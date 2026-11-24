import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

# Benchmarked Nov 2026: Groq primary for sub-2s latency, Gemini Flash-Lite cross-provider backup
MODEL_CHAINS = {
    "low": [
        ("groq", "openai/gpt-oss-20b"),
        ("groq", "qwen/qwen3.8-27b"),
        ("google", "gemini-3.5-flash-lite"),
    ],
    "medium": [
        ("groq", "openai/gpt-oss-120b"),
        ("groq", "qwen/qwen3.8-27b"),
        ("google", "gemini-3.5-flash-lite"),
    ],
    "high": [
        ("groq", "openai/gpt-oss-120b"),
        ("groq", "openai/gpt-oss-20b"),
        ("google", "gemini-3.5-flash-lite"),
    ],
}


def _build_llm(provider: str, model_name: str, groq_key: str | None, google_key: str | None):
    if provider == "groq" and groq_key:
        return ChatGroq(
            model=model_name,
            temperature=0,
            api_key=groq_key,
            timeout=10,
            max_retries=0,
        )
    if provider == "google" and google_key:
        # gemini-3.5-flash-lite uses fixed sampling defaults (omit temperature to avoid warnings)
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_key,
            timeout=10,
            max_retries=0,
        )
    return None


@lru_cache(maxsize=4)
def pick_llm(level: str):
    level = level.lower()
    if level not in MODEL_CHAINS:
        raise ValueError(f"Unsupported level: {level}")

    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    llms = []
    for provider, model_name in MODEL_CHAINS[level]:
        client = _build_llm(provider, model_name, groq_key, google_key)
        if client is not None:
            llms.append(client)

    if not llms:
        raise RuntimeError("No API keys found — set GROQ_API_KEY and/or GOOGLE_API_KEY in .env")

    return llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]