"""
llm_fallback.py
---------------
Sequential LLM fallback chain for the RAG pipeline.
Order: Groq → Cerebras → Mistral → SambaNova → GitHub/Phi-4

Each provider is tried in order. If one raises an exception or returns an
empty response it is logged and the next provider is attempted.
The function signature mirrors what RAGPipeline.query() expects:
    call_llm(messages, temperature, max_tokens) → str
"""

import os
import time
import logging
from typing import Optional

from colorama import Fore, Style

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  RETRY DECORATOR  (mirrors Embedder._request pattern)
# ══════════════════════════════════════════════════════════════════════════════
def _with_retry(fn, provider_name: str, max_attempts: int = 3):
    """
    Wraps a provider call with exponential-backoff retry on 429 / 503.
    Returns the result or raises the last exception.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            err_str  = str(e).lower()

            if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
                wait = 8 * (2 ** (attempt - 1))          # 8 → 16 → 32 s
                print(
                    f"{Fore.YELLOW}   [{provider_name}] rate-limited "
                    f"— retrying in {wait}s (attempt {attempt}/{max_attempts}){Style.RESET_ALL}"
                )
                time.sleep(wait)

            elif "503" in err_str or "service unavailable" in err_str:
                wait = 15
                print(
                    f"{Fore.YELLOW}   [{provider_name}] service unavailable "
                    f"— retrying in {wait}s (attempt {attempt}/{max_attempts}){Style.RESET_ALL}"
                )
                time.sleep(wait)

            else:
                # Non-retryable error — fail fast
                raise

    raise last_exc


# ══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL PROVIDER CALLERS
# ══════════════════════════════════════════════════════════════════════════════

def _call_groq(messages: list, temperature: float, max_tokens: int) -> str:
    import groq

    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    model  = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    def _do():
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return _with_retry(_do, "Groq")


def _call_cerebras(messages: list, temperature: float, max_tokens: int) -> str:
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])
    model  = os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b")

    def _do():
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return _with_retry(_do, "Cerebras")


def _call_mistral(messages: list, temperature: float, max_tokens: int) -> str:
    from mistralai import Mistral

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    model  = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")

    def _do():
        resp = client.chat.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return _with_retry(_do, "Mistral")


def _call_sambanova(messages: list, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["SAMBANOVA_API_KEY"],
        base_url="https://api.sambanova.ai/v1",
    )
    model = os.environ.get("SAMBANOVA_MODEL", "DeepSeek-V3.1")

    def _do():
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return _with_retry(_do, "SambaNova")


def _call_github(messages: list, temperature: float, max_tokens: int) -> str:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
    from azure.core.credentials import AzureKeyCredential

    client = ChatCompletionsClient(
        endpoint="https://models.github.ai/inference",
        credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
    )
    model = os.environ.get("GITHUB_MODEL", "microsoft/Phi-4")

    # Convert OpenAI-style message dicts → azure SDK message objects
    def _to_azure(msg: dict):
        role    = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            return SystemMessage(content)
        if role == "assistant":
            return AssistantMessage(content)
        return UserMessage(content)

    azure_messages = [_to_azure(m) for m in messages]

    def _do():
        resp = client.complete(
            messages=azure_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return _with_retry(_do, "GitHub/Phi-4")


# ══════════════════════════════════════════════════════════════════════════════
#  FALLBACK CHAIN
# ══════════════════════════════════════════════════════════════════════════════

# Ordered list of (display_name, required_env_var, caller_fn)
_PROVIDERS = [
    ("Groq",          "GROQ_API_KEY",       _call_groq),
    ("Cerebras",      "CEREBRAS_API_KEY",    _call_cerebras),
    ("Mistral",       "MISTRAL_API_KEY",     _call_mistral),
    ("SambaNova",     "SAMBANOVA_API_KEY",   _call_sambanova),
    ("GitHub/Phi-4",  "GITHUB_TOKEN",        _call_github),
]


def call_llm(
    messages:    list,
    temperature: float = 0.3,
    max_tokens:  int   = 1024,
) -> tuple[str, str]:
    """
    Try each provider in order. Returns (response_text, provider_name).
    Raises RuntimeError if every provider fails.

    Usage:
        answer, provider = call_llm(messages, temperature=0.3, max_tokens=1024)
    """
    errors: list[str] = []

    for name, env_var, caller in _PROVIDERS:
        # Skip providers whose API key is not configured
        if not os.environ.get(env_var):
            print(f"{Fore.YELLOW}⚠  [{name}] skipped — {env_var} not set{Style.RESET_ALL}")
            errors.append(f"{name}: env var {env_var} missing")
            continue

        print(f"{Fore.CYAN}→  Trying {name}...{Style.RESET_ALL}", end=" ", flush=True)
        try:
            result = caller(messages, temperature, max_tokens)
            if result and result.strip():
                print(f"{Fore.GREEN}✔{Style.RESET_ALL}")
                return result, name
            else:
                print(f"{Fore.YELLOW}empty response{Style.RESET_ALL}")
                errors.append(f"{name}: empty response")

        except Exception as e:
            print(f"{Fore.RED}✗ {e}{Style.RESET_ALL}")
            errors.append(f"{name}: {e}")
            logger.warning("[%s] failed: %s", name, e)

    raise RuntimeError(
        "All LLM providers failed.\n" + "\n".join(f"  • {e}" for e in errors)
    )