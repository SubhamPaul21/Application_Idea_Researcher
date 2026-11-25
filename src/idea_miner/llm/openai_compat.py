"""Any OpenAI-compatible chat-completions endpoint: OpenAI, OpenRouter, Groq, Gemini, DeepSeek,
Mistral, Together, Ollama, LM Studio, vLLM... JSON mode degrades gracefully per provider:
json_schema -> json_object -> plain prompt instructions."""
from __future__ import annotations

import json
import re
import time
from typing import Any

import openai
from openai import OpenAI

from .base import LLMClient, LLMError, LLMResult

_JSON_HINT = ("\n\nRespond with a single JSON object (no prose, no code fences) that conforms to this "
              "JSON Schema:\n{schema}")


def _retry_after(message: str, attempt: int) -> float:
    """Read 'try again in 21.375s' / '90ms' / '1m2s' from a rate-limit message; else back off exponentially."""
    m = re.search(r"try again in (?:(\d+)m)?(?:([\d.]+)(ms|s))?", message)
    if m and (m.group(1) or m.group(2)):
        secs = int(m.group(1) or 0) * 60
        if m.group(2):
            secs += float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
        return min(secs + 1.0, 90.0)
    return min(5.0 * attempt, 60.0)


class OpenAICompatLLM(LLMClient):
    def __init__(self, provider: str, model: str, *, base_url: str, api_key: str | None = None,
                 json_mode: str = "json_object", max_tokens_param: str = "max_tokens") -> None:
        super().__init__(model)
        self.provider = provider
        self.json_mode = json_mode  # json_schema | json_object | none
        self.max_tokens_param = max_tokens_param
        headers = {"X-Title": "idea-miner"} if provider == "openrouter" else None
        self.client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, max_retries=4,
                             timeout=300, default_headers=headers)
        if provider != "openai":
            # The SDK also reads OPENAI_ORG_ID, OPENAI_PROJECT_ID and OPENAI_CUSTOM_HEADERS from the environment and
            # sends them to whatever base URL it is given. Those belong to OpenAI only: drop them for other providers.
            self.client.organization = None
            self.client.project = None
            if hasattr(self.client, "_custom_headers"):
                self.client._custom_headers = dict(headers or {})
            if hasattr(self.client, "_ambient_authorizations"):
                self.client._ambient_authorizations = frozenset()

    def _build(self, system: str, user: str, max_tokens: int,
               schema: dict[str, Any] | None) -> tuple[list[dict[str, str]], dict[str, Any]]:
        kwargs: dict[str, Any] = {self.max_tokens_param: max_tokens}
        if schema is not None:
            system = system + "\n\nAlways answer with valid JSON only."
            if self.json_mode == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": schema, "strict": True},
                }
            else:
                user = user + _JSON_HINT.format(schema=json.dumps(schema))
                if self.json_mode == "json_object":
                    kwargs["response_format"] = {"type": "json_object"}
        return [{"role": "system", "content": system}, {"role": "user", "content": user}], kwargs

    def _adapt(self, message: str, wants_json: bool) -> bool:
        """Adjust request shape after a 400 from a provider that doesn't support something."""
        if "max_completion_tokens" in message and self.max_tokens_param == "max_tokens":
            self.max_tokens_param = "max_completion_tokens"
            return True
        if "max_completion_tokens" not in message and "max_tokens" in message \
                and self.max_tokens_param == "max_completion_tokens":
            self.max_tokens_param = "max_tokens"
            return True
        if wants_json and ("response_format" in message or "json" in message):
            if self.json_mode == "json_schema":
                self.json_mode = "json_object"
                return True
            if self.json_mode == "json_object":
                self.json_mode = "none"
                return True
        return False

    def _complete(self, system: str, user: str, *, max_tokens: int, task: str,
                  schema: dict[str, Any] | None) -> LLMResult:
        resp = None
        rate_waits = 0
        for _ in range(12):
            messages, kwargs = self._build(system, user, max_tokens, schema)
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
                break
            except openai.RateLimitError as e:
                rate_waits += 1
                msg = str(e).lower()
                # A per-day / quota limit won't clear in a minute: fail fast with the provider's own message.
                if rate_waits > 6 or "quota" in msg or "per day" in msg or "(tpd)" in msg or "(rpd)" in msg:
                    raise LLMError(f"{self.name}: {e}") from e
                time.sleep(_retry_after(str(e), rate_waits))
            except openai.BadRequestError as e:
                if not self._adapt(str(e).lower(), schema is not None):
                    raise LLMError(f"{self.name}: {e}") from e
            except openai.APIError as e:
                raise LLMError(f"{self.name}: {e}") from e
        if resp is None:
            raise LLMError(f"{self.name}: could not find request parameters this provider accepts")
        if not resp.choices:
            raise LLMError(f"{self.name}: empty response")
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResult(
            text=choice.message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            truncated=choice.finish_reason == "length",
        )

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self.client.models.list())
        except openai.APIError as e:
            raise LLMError(f"{self.name}: {e}") from e
