"""
Build the gate's training set:
  ~500 in-domain query variants (via Groq) across the 3 Phase-1 intents,
  ~500 out-of-domain queries sampled from SQuAD 2.

Writes data/gate_train.jsonl.
"""
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from groq import Groq, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from civsa.llm_client import chat as llm_chat

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

client = Groq()   # reads GROQ_API_KEY from env

# Model to use for variant generation. 70B gives best diversity;
# 8B-instant is fine and faster/cheaper if you're re-running often.
MODEL     = "openai/gpt-oss-120b"   # in-domain paraphrasings — quality matters
OOD_MODEL = "openai/gpt-oss-20b"    # off-topic — easy task, higher rate limit

SEED_INTENTS = {
    "vendor_attribute": (
        "Ask what a specific vendor has quoted for a specific item. "
        "Include realistic Indian vendor names like 'Rajshree Fasteners', "
        "'Nirmala Chemicals', 'Bharat Traders', 'ABC Industries', 'XYZ Ltd', "
        "'Precision Tools'. Include realistic item names like "
        "'M10 x 40 hex bolt', 'stainless steel washer', 'PTFE gasket', "
        "'acetone drums', 'safety helmet', 'IS 2925 certified helmet'. "
        "Vary between very short queries (2-4 words) and long polite ones."
    ),
    "cert_filter": (
        "Ask which of the buyer's vendors have a particular certification. "
        "Reference certifications like ISO 9001, ISO 14001, BIS, MSME, "
        "IATF 16949. Vary between filter-style and list-style queries."
    ),
    "price_compare": (
        "Ask to compare quoted prices for the same item across vendors. "
        "Reference item names, dates ('last week', 'this month'), and "
        "sometimes ask for the cheapest or best-value option."
    ),
}


def call_with_retry(fn, max_retries: int = 5):
    """
    Call fn(), retrying on Groq rate limits. Groq's error message includes
    'try again in Xs', so we parse that hint and wait accordingly.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError as e:
            m = re.search(r"try again in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 1 if m else 2 ** attempt
            print(f"  · rate-limited, waiting {wait:.1f}s "
                  f"(attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError("Rate-limit retries exhausted")




def generate_intent(name: str, brief: str,
                    n: int = 120, batch_size: int = 30) -> list[dict]:
    """
    Generate n queries in batches of batch_size. Small batches are much more
    reliable than one huge batch — the model rarely stops early on 30 items
    but often does on 120.
    """
    collected: list[str] = []
    batches = (n + batch_size - 1) // batch_size

    for b in range(batches):
        remaining = min(batch_size, n - len(collected))
        if remaining <= 0:
            break

        prompt = (
            f"Give me {remaining} different natural phrasings of a procurement "
            f"buyer's query in the intent category '{name}': {brief}\n"
            "Vary formality, length, word order, and include some typos.\n"
            "Return ONE query per line. NO numbering. NO bullets. NO headings. "
            "NO markdown formatting. NO preamble. Just the questions, one per line."
        )

        text = call_with_retry(lambda: llm_chat(
            messages=[{"role": "user", "content": prompt}],
            model_size="fast",           # gate data is bulk classification-style — small model is fine
            temperature=1.0,
            max_tokens=2500,
        ))

        lines = _parse_lines(text)
        if not lines:
            # Show the first 300 chars so we can see what the model actually returned
            print(f"  · WARN {name} batch {b+1}: 0 lines parsed. Raw: {text[:300]!r}",
                  flush=True)
        collected.extend(lines[:remaining])
        print(f"  · {name} batch {b+1}/{batches}: +{len(lines[:remaining])} "
              f"(total {len(collected)}/{n})", flush=True)

        time.sleep(8)

    return [{"text": t, "label": 1, "intent": name} for t in collected]


def _parse_lines(text: str) -> list[str]:
    """
    Robust line parsing that handles:
      - '1. Query text'
      - '12) Query text'
      - '- Query text'
      - '* Query text'
      - '**Query text**' (markdown bold)
      - Surrounding quotes
    """
    out = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # Strip leading numbering: "1.", "12.", "1)", "12)"
        s = re.sub(r"^\d+[.\)]\s*", "", s)
        # Strip leading bullets and surrounding whitespace
        s = s.strip("-•*\t ")
        # Strip surrounding markdown bold **...**
        s = re.sub(r"^\*\*(.+?)\*\*$", r"\1", s)
        # Strip surrounding quotes
        s = s.strip('"\'')
        # Reject too-short lines (usually preamble fragments)
        if len(s) > 5:
            out.append(s)
    return out





OOD_CATEGORIES = [
    "general knowledge (history, science, geography)",
    "weather, time and dates",
    "personal chit-chat and greetings",
    "programming help unrelated to procurement",
    "cooking, food and recipes",
    "travel, directions, and transport",
    "entertainment (movies, music, sports)",
    "health, fitness and wellness",
    "philosophy and abstract musings",
    "jokes and casual banter",
]

def out_of_domain(n: int = 500) -> list[dict]:
    """Generate diverse off-topic queries using the smaller OOD model."""
    per_cat = max(1, n // len(OOD_CATEGORIES))
    rows: list[dict] = []
    for cat in OOD_CATEGORIES:
        prompt = (
            f"Give me {per_cat} different everyday questions people ask "
            f"in the category: {cat}. "
            "Return ONE question per line. No numbering, no bullets, no headings."
        )
        text = call_with_retry(lambda: llm_chat(
            messages=[{"role": "user", "content": prompt}],
            model_size="fast",
            temperature=1.0,
            max_tokens=1500,
        ))
        lines = [l.strip("-•* \t") for l in text.splitlines()
                 if l.strip() and len(l.strip()) > 5]
        rows.extend({"text": l, "label": 0, "intent": "off_topic"}
                    for l in lines[:per_cat])
        time.sleep(8)
    return rows


def main():
    random.seed(42)
    rows = []
    for name, brief in SEED_INTENTS.items():
        rows.extend(generate_intent(name, brief))
        print(f"  · generated {len([r for r in rows if r['intent']==name])} "
              f"queries for '{name}'")
    rows.extend(out_of_domain())
    print(f"  · added {sum(1 for r in rows if r['label']==0)} "
          f"out-of-domain queries")

    out = DATA_DIR / "gate_train.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()