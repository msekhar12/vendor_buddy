"""
Unified LLM client with provider fallback.

Tries providers in the order defined by LLM_PROVIDER_ORDER in config.py.
First provider to return a response wins. A provider only "fails over"
on genuine errors (network down, model not pulled, API key missing,
rate limits) — an empty-but-successful response is treated as a real
answer, not a failure.
"""
import os

from .config import (
    OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_MODEL_FAST,
    GROQ_MODEL, GROQ_MODEL_FAST,
    LLM_PROVIDER_ORDER,
)


def chat(messages: list[dict],
         model_size: str = "fast",
         temperature: float = 0.0,
         max_tokens: int = 400) -> str:
    """
    Send a chat completion request. Tries providers in order.
    Returns the response text, or raises RuntimeError if all fail.

    model_size: "fast" (small model for classification-type tasks)
                "big"  (bigger model for RAG answers)
    """
    errors: list[str] = []
    for provider in LLM_PROVIDER_ORDER:
        try:
            if provider == "ollama":
                return _call_ollama(messages, model_size, temperature, max_tokens)
            if provider == "groq":
                return _call_groq(messages, model_size, temperature, max_tokens)
            print(f"[llm_client] unknown provider: {provider}", flush=True)
        except Exception as e:
            print(f"[llm_client] {provider} failed: {e} — trying next",
                  flush=True)
            errors.append(f"{provider}: {e}")
            continue
    raise RuntimeError(f"All LLM providers failed: {errors}")


def _call_ollama(messages, model_size, temperature, max_tokens) -> str:
    import ollama

    model = OLLAMA_MODEL_FAST if model_size == "fast" else OLLAMA_MODEL
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    print(f"[llm_client] ollama · {model} · ok", flush=True)
    return resp["message"]["content"] or ""


def _call_groq(messages, model_size, temperature, max_tokens) -> str:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set — cannot use Groq fallback")
    from groq import Groq

    model = GROQ_MODEL_FAST if model_size == "fast" else GROQ_MODEL
    client = Groq()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print(f"[llm_client] groq · {model} · ok (fallback)", flush=True)
    return resp.choices[0].message.content or ""