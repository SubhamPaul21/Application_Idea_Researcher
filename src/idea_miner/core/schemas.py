"""Structured outputs the LLM must return. Validation is lenient so that any provider works:
enum values are normalized, unknown values fall back to safe defaults, numbers are clamped."""
from __future__ import annotations

from typing import Any, ClassVar, Literal, get_args, get_origin

from pydantic import BaseModel, field_validator, model_validator

AppType = Literal[
    "script", "cli", "web_app", "browser_extension", "mobile_app",
    "desktop_app", "api_service", "bot_or_integration", "not_software",
]
Audience = Literal["consumers", "professionals", "small_businesses", "developers", "enterprises"]
Frequency = Literal["daily", "weekly", "monthly", "rare", "unknown"]
WTP = Literal["explicit", "implied", "none"]
Effort = Literal["weekend", "weeks", "months"]
Verdict = Literal["unsolved", "partially_solved", "solved"]

APP_TYPES: list[str] = list(get_args(AppType))
SOFTWARE_APP_TYPES: list[str] = [a for a in APP_TYPES if a != "not_software"]
AUDIENCES: list[str] = list(get_args(Audience))
EFFORTS: list[str] = list(get_args(Effort))


class LenientModel(BaseModel):
    FALLBACKS: ClassVar[dict[str, Any]] = {}

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for name, field in cls.model_fields.items():
            value = data.get(name)
            if value is None:
                if name in cls.FALLBACKS:
                    data[name] = cls.FALLBACKS[name]
                continue
            ann = field.annotation
            if get_origin(ann) is Literal:
                allowed = get_args(ann)
                norm = str(value).strip().lower().replace(" ", "_").replace("-", "_")
                data[name] = norm if norm in allowed else cls.FALLBACKS.get(name, allowed[0])
            elif ann is bool and not isinstance(value, bool):
                data[name] = str(value).strip().lower() in ("true", "yes", "y", "1")
            elif ann is int and not isinstance(value, int):
                try:
                    data[name] = int(float(value))
                except (TypeError, ValueError):
                    data[name] = cls.FALLBACKS.get(name, 0)
            elif ann == list[str]:
                if isinstance(value, str):
                    data[name] = [value] if value.strip() else []
                elif isinstance(value, list):
                    data[name] = [str(v) for v in value if v not in (None, "")]
            elif ann is str and not isinstance(value, str):
                data[name] = str(value)
        return data


class ProblemCard(LenientModel):
    FALLBACKS: ClassVar[dict[str, Any]] = {
        "audience": "consumers", "frequency": "unknown", "willingness_to_pay": "none",
        "app_type": "web_app", "build_effort": "weeks", "severity": 1,
        "existing_solutions": [], "who_has_it": "unknown", "problem_statement": "",
        "is_problem": False, "software_solvable": False,
    }

    item_id: str
    is_problem: bool
    problem_statement: str
    who_has_it: str
    audience: Audience
    frequency: Frequency
    severity: int
    willingness_to_pay: WTP
    existing_solutions: list[str]
    software_solvable: bool
    app_type: AppType
    build_effort: Effort

    @field_validator("severity")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return max(1, min(10, v))


class ClassificationBatch(LenientModel):
    results: list[ProblemCard]


class ClusterSummary(LenientModel):
    FALLBACKS: ClassVar[dict[str, Any]] = {
        "app_type": "web_app", "audience": "consumers", "build_effort": "weeks", "search_terms": [],
        "target_user": "", "app_concept": "",
    }

    title: str
    problem: str
    target_user: str
    app_concept: str
    app_type: AppType
    audience: Audience
    build_effort: Effort
    search_terms: list[str]


class Alternative(LenientModel):
    name: str
    url: str
    gap: str


class ExistenceVerdict(LenientModel):
    FALLBACKS: ClassVar[dict[str, Any]] = {"verdict": "partially_solved", "closest_alternatives": [], "opportunity": ""}

    verdict: Verdict
    closest_alternatives: list[Alternative]
    opportunity: str
