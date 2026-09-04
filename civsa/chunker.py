"""
Paragraph-based chunking with a small sliding sentence window for overlap.

Split on blank lines first (paragraphs), then apply a 3-sentence window with
1-sentence overlap so no meaning is lost at chunk boundaries.
"""
import re

# Sentence splitter that respects common terminators.
_SENT = re.compile(r"(?<=[\.\!\?])\s+")


def chunk_paragraphs(text: str, sentences_per_chunk: int = 3,
                     overlap_sentences: int = 1) -> list[dict]:
    """
    Returns a list of {"text": ..., "para_index": int, "chunk_index": int}.
    para_index lets us cite back to a specific paragraph in the source doc.
    """
    chunks = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunk_idx = 0

    for pi, para in enumerate(paragraphs):
        sents = _SENT.split(para)
        step = max(1, sentences_per_chunk - overlap_sentences)
        for s_start in range(0, len(sents), step):
            window = sents[s_start:s_start + sentences_per_chunk]
            chunk_text = " ".join(window).strip()
            if len(chunk_text) < 20:      # skip trivially short chunks
                continue
            chunks.append({
                "text":        chunk_text,
                "para_index":  pi,
                "chunk_index": chunk_idx,
            })
            chunk_idx += 1
    return chunks