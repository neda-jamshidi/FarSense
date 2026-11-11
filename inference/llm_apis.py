"""One `chat()` call for the five API models used in the paper.

API keys are read from environment variables (or a `.env` file in the repo root):
OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, QWEN_API_KEY, MISTRAL_API_KEY.
"""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# short name -> (provider, default model id)
MODELS = {
    "gpt-4o": ("openai", "gpt-4o"),
    "gemini-2.5-flash": ("gemini", "gemini-2.5-flash"),
    "deepseek-chat": ("deepseek", "deepseek-chat"),
    "qwen-plus": ("qwen", "qwen-plus"),
    "mistral-large": ("mistral", "mistral-large-latest"),
}

_OPENAI_COMPATIBLE = {
    "openai": (None, "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "QWEN_API_KEY"),
}


def _key(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} in your environment or in .env (see .env.example).")
    return value


def chat(model: str, user: str, system: str | None = None, model_id: str | None = None) -> str:
    """Send one user message (plus an optional system prompt) and return the reply text."""
    provider, default_id = MODELS[model]
    model_id = model_id or default_id
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]

    if provider in _OPENAI_COMPATIBLE:
        from openai import OpenAI

        base_url, key_name = _OPENAI_COMPATIBLE[provider]
        client = OpenAI(api_key=_key(key_name), base_url=base_url)
        resp = client.chat.completions.create(model=model_id, messages=messages)
        return resp.choices[0].message.content

    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=_key("GEMINI_API_KEY"))
        config = types.GenerateContentConfig(system_instruction=system) if system else None
        resp = client.models.generate_content(model=model_id, contents=user, config=config)
        return resp.text

    if provider == "mistral":
        from mistralai import Mistral

        client = Mistral(api_key=_key("MISTRAL_API_KEY"))
        resp = client.chat.complete(model=model_id, messages=messages)
        return resp.choices[0].message.content

    raise ValueError(f"Unknown provider: {provider}")
