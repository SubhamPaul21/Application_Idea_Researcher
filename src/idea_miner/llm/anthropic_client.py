"""Claude via the official Anthropic SDK: native structured outputs (output_config.format),
per-task effort, prompt caching, and server-side refusal fallbacks on models that support them."""
from __future__ import annotations

from typing import Any

import anthropic

from .base import LLMClient, LLMError, LLMRefusal, LLMResult

# Models that accept output_config.effort
_EFFORT_PREFIXES = (
    "claude-opus-5", "claude-sonnet-5", "claude-haiku-5", "claude-fable", "claude-mythos",
    "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6",
)
# Models that accept the server-side `fallbacks: "default"` parameter
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


DEFAULT_BASE_URL = "https://api.anthropic.com"


class AnthropicLLM(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, *, api_key: str | None = None, effort: dict[str, str] | None = None,
                 refusal_fallbacks: bool = True, base_url: str = DEFAULT_BASE_URL) -> None:
        super().__init__(model)
        if not api_key:
            # Fail closed: without an explicit key the SDK would fall back to ambient credentials
            # (ANTHROPIC_AUTH_TOKEN, a saved profile, workload identity) that this app never asked for.
            raise LLMError("No Anthropic API key: set ANTHROPIC_API_KEY in .env or paste a key in the sidebar.")
        # Explicit api_key and base_url: the SDK then ignores credential env vars and ANTHROPIC_BASE_URL.
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, max_retries=4)
        self.effort = effort or {}
        self.use_fallbacks = refusal_fallbacks and model in _FALLBACK_MODELS

    def _complete(self, system: str, user: str, *, max_tokens: int, task: str,
                  schema: dict[str, Any] | None) -> LLMResult:
        output_config: dict[str, Any] = {}
        if self.model.startswith(_EFFORT_PREFIXES):
            output_config["effort"] = self.effort.get(task, "medium")
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "cache_control": {"type": "ephemeral"},  # caches the shared system-prompt prefix
        }
        if output_config:
            params["output_config"] = output_config

        try:
            resp = self._send(params)
        except anthropic.APIError as e:
            raise LLMError(f"{self.name}: {e}") from e

        if resp.stop_reason == "refusal":
            category = getattr(getattr(resp, "stop_details", None), "category", None)
            raise LLMRefusal(f"{self.name} declined the request (category: {category})")
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = resp.usage
        return LLMResult(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            truncated=resp.stop_reason == "max_tokens",
            served_model=getattr(resp, "model", None),
        )

    def _send(self, params: dict[str, Any]) -> Any:
        if self.use_fallbacks:
            try:
                return self.client.beta.messages.create(betas=[_FALLBACK_BETA], fallbacks="default", **params)
            except TypeError:
                # SDK predates the `fallbacks` kwarg: pass it through the raw body instead
                return self.client.beta.messages.create(
                    betas=[_FALLBACK_BETA], extra_body={"fallbacks": "default"}, **params
                )
            except anthropic.BadRequestError as e:
                if "fallback" not in str(e).lower():
                    raise
                self.use_fallbacks = False  # not available for this account/model: stop trying
        return self.client.messages.create(**params)

    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list()]
