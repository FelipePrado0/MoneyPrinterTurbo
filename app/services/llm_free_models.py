"""Ordered, self-healing list of OpenRouter models to try for one request.

OpenRouter's free catalog changes without notice: a ``:free`` slug can be
withdrawn (404) or throttled upstream (429) at any time, and a hardcoded
model then breaks every LLM call in the app. This module builds the attempt
order from the live ``/models`` catalog instead: the configured model first,
then the remaining general-purpose free models, then OpenRouter's own
``openrouter/free`` router as the last resort. Failures reported by the
caller temporarily remove a model from the list.
"""

import re
import threading
import time

import requests
from loguru import logger

from app.models.llm_provider import get_llm_provider

MODELS_URL = "https://openrouter.ai/api/v1/models"
FREE_ROUTER_MODEL = "openrouter/free"
CATALOG_TTL_SECONDS = 6 * 3600
RATE_LIMIT_COOLDOWN_SECONDS = 600
CATALOG_TIMEOUT_SECONDS = 15
MIN_CONTEXT_LENGTH = 32768
MIN_PARAMETERS_BILLIONS = 20

# ponytail: substring blocklist for specialised/small models; swap for a
# quality ranking if a bad model still slips through.
_EXCLUDED_MARKERS = (
    "code",
    "coder",
    "safety",
    "guard",
    "sante",
    "-fin",
    "mini",
    "nano",
    "small",
    "-xs",
    "tiny",
)
_PARAMETERS_RE = re.compile(r"(\d+(?:\.\d+)?)b\b")

_lock = threading.Lock()
_catalog: list[dict] = []
_catalog_fetched_at: float | None = None
_gone: set[str] = set()
_cooldown_until: dict[str, float] = {}


def preferred_model(app_config) -> str:
    """The configured OpenRouter model; an empty value means the provider
    default (the WebUI stores "" when the field equals the default)."""
    return get_llm_provider("openrouter").resolve_model_name(
        str(app_config.get("openrouter_model_name", "") or "")
    )


def _now() -> float:
    return time.time()


def reset_state() -> None:
    global _catalog, _catalog_fetched_at
    with _lock:
        _catalog = []
        _catalog_fetched_at = None
        _gone.clear()
        _cooldown_until.clear()


def _fetch_catalog() -> list[dict]:
    response = requests.get(MODELS_URL, timeout=CATALOG_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("data", [])


def _is_zero_price(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    return str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0"


def _is_eligible_free_model(model: dict) -> bool:
    model_id = str(model.get("id", ""))
    if not model_id.endswith(":free") or not _is_zero_price(model):
        return False
    modalities = (model.get("architecture") or {}).get("output_modalities") or ["text"]
    if modalities != ["text"]:
        return False
    if int(model.get("context_length") or 0) < MIN_CONTEXT_LENGTH:
        return False
    lowered = model_id.lower()
    if any(marker in lowered for marker in _EXCLUDED_MARKERS):
        return False
    size = _PARAMETERS_RE.search(lowered)
    if size and float(size.group(1)) < MIN_PARAMETERS_BILLIONS:
        return False
    return True


def _refresh_catalog_locked() -> None:
    global _catalog, _catalog_fetched_at
    if _catalog_fetched_at is not None and (
        _now() - _catalog_fetched_at < CATALOG_TTL_SECONDS
    ):
        return
    # A re-fetch is the only point where a withdrawn model may come back.
    if _catalog_fetched_at is not None:
        _gone.clear()
    _catalog_fetched_at = _now()
    try:
        _catalog = _fetch_catalog()
    except Exception as exc:
        logger.warning(f"failed to refresh OpenRouter model catalog: {exc}")


def _is_free_id(model_id: str) -> bool:
    if model_id == FREE_ROUTER_MODEL or model_id.endswith(":free"):
        return True
    return any(m.get("id") == model_id and _is_zero_price(m) for m in _catalog)


def _ordered_models_locked(preferred: str, free_only: bool) -> list[str]:
    _refresh_catalog_locked()
    ordered: list[str] = []
    if preferred and (not free_only or _is_free_id(preferred)):
        ordered.append(preferred)
    free_models = sorted(
        (m for m in _catalog if _is_eligible_free_model(m)),
        key=lambda m: -int(m.get("context_length") or 0),
    )
    ordered += [m["id"] for m in free_models if m["id"] not in ordered]
    if FREE_ROUTER_MODEL not in ordered:
        ordered.append(FREE_ROUTER_MODEL)
    return ordered


def _state_locked(model_id: str) -> str:
    if model_id in _gone:
        return "unavailable"
    if _cooldown_until.get(model_id, 0) > _now():
        return "cooldown"
    return "ok"


def candidates(preferred: str, free_only: bool) -> list[str]:
    """Models to try, in order, skipping ones currently known to fail."""
    with _lock:
        return [
            model_id
            for model_id in _ordered_models_locked(preferred, free_only)
            if _state_locked(model_id) == "ok"
        ]


def report_failure(model_id: str, kind: str) -> None:
    """``gone`` (404/403) drops the model until the next catalog refresh;
    ``rate_limited`` (429) cools it down for a few minutes."""
    with _lock:
        if kind == "gone":
            _gone.add(model_id)
        elif kind == "rate_limited":
            _cooldown_until[model_id] = _now() + RATE_LIMIT_COOLDOWN_SECONDS


def status(preferred: str, free_only: bool) -> list[dict]:
    """Every model in attempt order with its current state, for the UI/API."""
    with _lock:
        ordered = _ordered_models_locked(preferred, free_only)
        contexts = {m.get("id"): m.get("context_length") for m in _catalog}
        return [
            {
                "id": model_id,
                "context_length": contexts.get(model_id),
                "state": _state_locked(model_id),
                "cooldown_until": _cooldown_until.get(model_id)
                if _state_locked(model_id) == "cooldown"
                else None,
            }
            for model_id in ordered
        ]
