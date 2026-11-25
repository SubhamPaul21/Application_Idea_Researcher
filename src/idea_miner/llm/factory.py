"""Provider presets and the single entry point `make_llm`."""
from __future__ import annotations

from typing import Any

from .. import config
from ..core import db
from .base import LLMClient, LLMError, LLMResult

PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "label": "Anthropic (Claude)", "kind": "anthropic", "key_env": "ANTHROPIC_API_KEY",
        "models": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
    },
    "openai": {
        "label": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY", "json_mode": "json_schema", "max_tokens_param": "max_completion_tokens",
    },
    "openrouter": {
        "label": "OpenRouter", "kind": "openai", "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY", "json_mode": "json_object",
    },
    "groq": {
        "label": "Groq", "kind": "openai", "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY", "json_mode": "json_object",
    },
    "gemini": {
        "label": "Google Gemini (OpenAI-compatible)", "kind": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY", "json_mode": "json_object",
    },
    "deepseek": {
        "label": "DeepSeek", "kind": "openai", "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY", "json_mode": "json_object",
    },
    "mistral": {
        "label": "Mistral", "kind": "openai", "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY", "json_mode": "json_object",
    },
    "together": {
        "label": "Together AI", "kind": "openai", "base_url": "https://api.together.xyz/v1",
        "key_env": "TOGETHER_API_KEY", "json_mode": "json_object",
    },
    "ollama": {
        "label": "Ollama (local)", "kind": "openai", "base_url": "http://localhost:11434/v1",
        "key_env": None, "json_mode": "json_object",
    },
    "lmstudio": {
        "label": "LM Studio (local)", "kind": "openai", "base_url": "http://localhost:1234/v1",
        "key_env": None, "json_mode": "none",
    },
    "custom": {
        "label": "Custom OpenAI-compatible endpoint", "kind": "openai", "base_url_env": "CUSTOM_LLM_BASE_URL",
        "key_env": "CUSTOM_LLM_API_KEY", "json_mode": "json_object",
    },
}


def resolve_key(provider: str, override: str | None = None) -> str | None:
    if override:
        return override
    return config.env(PROVIDERS[provider].get("key_env"))


def key_available(provider: str, override: str | None = None) -> bool:
    preset = PROVIDERS[provider]
    if not preset.get("key_env"):
        return True  # local servers
    return bool(resolve_key(provider, override))


def default_model(provider: str) -> str:
    preset = PROVIDERS[provider]
    from_env = config.env(f"{provider.upper()}_MODEL")
    if from_env:
        return from_env
    defaults = config.settings().get("llm", {}).get("defaults", {})
    for role in ("classifier", "writer"):
        d = defaults.get(role, {})
        if d.get("provider") == provider and d.get("model"):
            return d["model"]
    return (preset.get("models") or [""])[0]


def default_base_url(provider: str) -> str:
    preset = PROVIDERS[provider]
    return preset.get("base_url") or config.env(preset.get("base_url_env")) or ""


def make_llm(provider: str, model: str, *, api_key: str | None = None, base_url: str | None = None,
             role: str = "classifier", track_usage: bool = True) -> LLMClient:
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider '{provider}'")
    if not model:
        raise LLMError(f"Choose a model for {PROVIDERS[provider]['label']}")
    preset = PROVIDERS[provider]
    client: LLMClient
    if preset["kind"] == "anthropic":
        from .anthropic_client import AnthropicLLM

        a = config.settings().get("llm", {}).get("anthropic", {})
        client = AnthropicLLM(model, api_key=resolve_key(provider, api_key), effort=a.get("effort", {}),
                              refusal_fallbacks=a.get("refusal_fallbacks", True))
    else:
        from .openai_compat import OpenAICompatLLM

        url = base_url or default_base_url(provider)
        if not url:
            raise LLMError("Set a base URL for the custom endpoint (CUSTOM_LLM_BASE_URL)")
        client = OpenAICompatLLM(provider, model, base_url=url, api_key=resolve_key(provider, api_key),
                                 json_mode=preset.get("json_mode", "json_object"),
                                 max_tokens_param=preset.get("max_tokens_param", "max_tokens"))
    if track_usage:
        def _record(c: LLMClient, task: str, res: LLMResult) -> None:
            db.record_usage(role, task, c.provider, c.model, res.input_tokens, res.output_tokens,
                            res.cache_read_tokens)
        client.on_usage = _record
    return client
