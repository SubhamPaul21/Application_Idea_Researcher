"""Provider-agnostic LLM interface. Adapters implement `_complete`; JSON parsing, validation and
one repair retry live here so every provider behaves the same."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, get_origin

from pydantic import BaseModel, ValidationError

from .schema import strict_schema

M = TypeVar("M", bound=BaseModel)


class LLMError(RuntimeError):
    """Any provider failure the pipeline should log and move past."""


class LLMRefusal(LLMError):
    pass


class LLMTruncated(LLMError):
    pass


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    truncated: bool = False


UsageHook = Callable[["LLMClient", str, LLMResult], None]


class LLMClient(ABC):
    provider: str = ""

    def __init__(self, model: str) -> None:
        self.model = model
        self.on_usage: UsageHook | None = None

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"

    @abstractmethod
    def _complete(self, system: str, user: str, *, max_tokens: int, task: str,
                  schema: dict[str, Any] | None) -> LLMResult:
        """Send one request. `schema` is a strict JSON Schema when JSON output is wanted."""

    def list_models(self) -> list[str]:
        return []

    def _call(self, system: str, user: str, *, max_tokens: int, task: str,
              schema: dict[str, Any] | None) -> LLMResult:
        result = self._complete(system, user, max_tokens=max_tokens, task=task, schema=schema)
        if self.on_usage:
            try:
                self.on_usage(self, task, result)
            except Exception:  # usage logging must never break the pipeline
                pass
        return result

    def complete_text(self, system: str, user: str, *, max_tokens: int = 8000, task: str = "text") -> str:
        res = self._call(system, user, max_tokens=max_tokens, task=task, schema=None)
        text = res.text.strip()
        if res.truncated:
            text += "\n\n_(Output was cut off at the token limit.)_"
        return text

    def complete_json(self, system: str, user: str, model_cls: type[M], *, max_tokens: int = 4000,
                      task: str = "json") -> M:
        schema = strict_schema(model_cls)
        res = self._call(system, user, max_tokens=max_tokens, task=task, schema=schema)
        if res.truncated:
            raise LLMTruncated(f"{self.name} hit the token limit ({max_tokens})")
        try:
            return parse_model(res.text, model_cls)
        except ValueError as err:  # includes JSONDecodeError and pydantic ValidationError
            repair = (f"{user}\n\nYour previous reply could not be parsed ({_short(err)}). "
                      "Reply again with only the JSON object.")
            res2 = self._call(system, repair, max_tokens=max_tokens, task=task, schema=schema)
            if res2.truncated:
                raise LLMTruncated(f"{self.name} hit the token limit ({max_tokens})") from err
            try:
                return parse_model(res2.text, model_cls)
            except ValueError as err2:
                raise LLMError(f"{self.name} returned invalid JSON twice: {_short(err2)}") from err2


def _short(err: Exception, n: int = 300) -> str:
    if isinstance(err, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()[:5])[:n]
    return str(err)[:n]


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating reasoning tags, code fences and stray prose."""
    t = _THINK.sub("", text or "").strip()
    fence = _FENCE.search(t)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object found in reply")
    start = min(starts)
    end = max(t.rfind("}"), t.rfind("]"))
    if end <= start:
        raise ValueError("no complete JSON object found in reply")
    return json.loads(t[start : end + 1])


def parse_model(text: str, model_cls: type[M]) -> M:
    data = extract_json(text)
    fields = model_cls.model_fields
    if isinstance(data, list) and len(fields) == 1:
        (only_name, only_field), = fields.items()
        if get_origin(only_field.annotation) is list:
            data = {only_name: data}
    return model_cls.model_validate(data)
