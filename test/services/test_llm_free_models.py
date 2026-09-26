import types
import unittest
from unittest.mock import patch

import httpx
import openai

from app.config import config
from app.services import llm, llm_free_models


def _model(model_id, context=262144, prompt="0", completion="0", modality="text"):
    return {
        "id": model_id,
        "context_length": context,
        "pricing": {"prompt": prompt, "completion": completion},
        "architecture": {"output_modalities": [modality]},
    }


CATALOG = [
    _model("big/general-120b:free", context=262144),
    _model("huge/context-model:free", context=1000000),
    _model("tiny/lfm-2.6b:free", context=65536),
    _model("vendor/north-mini-code:free"),
    _model("nvidia/content-safety:free"),
    _model("inclusion/ling-flash-sante:free"),
    _model("short/context-model:free", context=16000),
    _model("paid/model", prompt="0.000001", completion="0.000002"),
    _model("audio/lyria:free", modality="audio"),
    _model("openrouter/free", context=200000),
]


def _status_error(cls, status):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls(message=f"status {status}", response=response, body=None)


class FreeModelSelectionTest(unittest.TestCase):
    def setUp(self):
        llm_free_models.reset_state()
        self.clock = [1000.0]
        patcher = patch.object(llm_free_models, "_now", lambda: self.clock[0])
        patcher.start()
        self.addCleanup(patcher.stop)
        fetch = patch.object(llm_free_models, "_fetch_catalog", return_value=CATALOG)
        self.fetch = fetch.start()
        self.addCleanup(fetch.stop)

    def test_candidates_put_preferred_first_then_filtered_free_models_then_router(self):
        candidates = llm_free_models.candidates("big/general-120b:free", free_only=True)
        self.assertEqual(
            candidates,
            [
                "big/general-120b:free",
                "huge/context-model:free",
                llm_free_models.FREE_ROUTER_MODEL,
            ],
        )

    def test_free_only_skips_paid_preferred_model(self):
        candidates = llm_free_models.candidates("paid/model", free_only=True)
        self.assertNotIn("paid/model", candidates)
        self.assertEqual(candidates[0], "huge/context-model:free")

    def test_paid_preferred_model_is_kept_when_free_only_is_off(self):
        candidates = llm_free_models.candidates("paid/model", free_only=False)
        self.assertEqual(candidates[0], "paid/model")

    def test_gone_model_is_dropped_until_next_catalog_refresh(self):
        llm_free_models.report_failure("huge/context-model:free", "gone")
        self.assertNotIn(
            "huge/context-model:free",
            llm_free_models.candidates("", free_only=True),
        )
        self.clock[0] += llm_free_models.CATALOG_TTL_SECONDS + 1
        self.assertIn(
            "huge/context-model:free",
            llm_free_models.candidates("", free_only=True),
        )

    def test_rate_limited_model_cools_down_then_returns(self):
        llm_free_models.report_failure("big/general-120b:free", "rate_limited")
        self.assertNotIn(
            "big/general-120b:free",
            llm_free_models.candidates("big/general-120b:free", free_only=True),
        )
        self.clock[0] += llm_free_models.RATE_LIMIT_COOLDOWN_SECONDS + 1
        self.assertEqual(
            llm_free_models.candidates("big/general-120b:free", free_only=True)[0],
            "big/general-120b:free",
        )

    def test_catalog_failure_keeps_preferred_and_router(self):
        self.fetch.side_effect = RuntimeError("network down")
        self.assertEqual(
            llm_free_models.candidates("big/general-120b:free", free_only=True),
            ["big/general-120b:free", llm_free_models.FREE_ROUTER_MODEL],
        )

    def test_status_on_cold_cache_includes_context_length(self):
        rows = {row["id"]: row for row in llm_free_models.status("", True)}
        self.assertEqual(rows["huge/context-model:free"]["context_length"], 1000000)

    def test_status_reports_state_per_model(self):
        llm_free_models.report_failure("huge/context-model:free", "rate_limited")
        status = {row["id"]: row["state"] for row in llm_free_models.status("", True)}
        self.assertEqual(status["huge/context-model:free"], "cooldown")
        self.assertEqual(status["big/general-120b:free"], "ok")


class OpenRouterFallbackTest(unittest.TestCase):
    def setUp(self):
        llm_free_models.reset_state()
        self.saved = dict(config.app)
        config.app.update(
            {
                "llm_provider": "openrouter",
                "openrouter_api_key": "or-key",
                "openrouter_base_url": "",
                "openrouter_model_name": "big/general-120b:free",
                "openrouter_free_fallback": True,
                "llm_free_only": True,
            }
        )
        fetch = patch.object(llm_free_models, "_fetch_catalog", return_value=CATALOG)
        fetch.start()
        self.addCleanup(fetch.stop)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.saved)

    def _run(self, behaviours):
        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs["model"])
                outcome = behaviours[kwargs["model"]]
                if isinstance(outcome, Exception):
                    raise outcome
                message = types.SimpleNamespace(content=outcome)
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )
        with patch.object(llm, "OpenAI", return_value=client):
            result = llm._generate_response("hi")
        return result, calls

    def test_falls_back_on_404_and_429_until_a_model_answers(self):
        result, calls = self._run(
            {
                "big/general-120b:free": _status_error(openai.NotFoundError, 404),
                "huge/context-model:free": _status_error(openai.RateLimitError, 429),
                llm_free_models.FREE_ROUTER_MODEL: "ok from router",
            }
        )
        self.assertEqual(result, "ok from router")
        self.assertEqual(
            calls,
            [
                "big/general-120b:free",
                "huge/context-model:free",
                llm_free_models.FREE_ROUTER_MODEL,
            ],
        )
        self.assertEqual(llm.get_last_used_model(), llm_free_models.FREE_ROUTER_MODEL)

    def test_empty_answer_and_server_error_also_fall_back(self):
        result, _ = self._run(
            {
                "big/general-120b:free": _status_error(openai.InternalServerError, 502),
                "huge/context-model:free": "",
                llm_free_models.FREE_ROUTER_MODEL: "fine",
            }
        )
        self.assertEqual(result, "fine")

    def test_all_models_failing_returns_error_without_raising(self):
        result, _ = self._run(
            {
                "big/general-120b:free": _status_error(openai.RateLimitError, 429),
                "huge/context-model:free": _status_error(openai.RateLimitError, 429),
                llm_free_models.FREE_ROUTER_MODEL: _status_error(
                    openai.RateLimitError, 429
                ),
            }
        )
        self.assertTrue(result.startswith("Error: "))

    def test_invalid_key_stops_immediately(self):
        result, calls = self._run(
            {"big/general-120b:free": _status_error(openai.AuthenticationError, 401)}
        )
        self.assertTrue(result.startswith("Error: "))
        self.assertEqual(calls, ["big/general-120b:free"])

    def test_model_gated_with_403_is_skipped_not_fatal(self):
        result, calls = self._run(
            {
                "big/general-120b:free": _status_error(openai.PermissionDeniedError, 403),
                "huge/context-model:free": "answer",
            }
        )
        self.assertEqual(result, "answer")
        self.assertNotIn(
            "big/general-120b:free",
            llm_free_models.candidates("big/general-120b:free", free_only=True),
        )

    def test_fallback_disabled_uses_only_configured_model(self):
        config.app["openrouter_free_fallback"] = False
        result, calls = self._run(
            {"big/general-120b:free": _status_error(openai.NotFoundError, 404)}
        )
        self.assertTrue(result.startswith("Error: "))
        self.assertEqual(calls, ["big/general-120b:free"])


if __name__ == "__main__":
    unittest.main()


def test_preferred_model_resolves_empty_config_to_provider_default():
    from app.models.llm_provider import get_llm_provider

    default = get_llm_provider("openrouter").default_model
    assert llm_free_models.preferred_model({"openrouter_model_name": ""}) == default
    assert llm_free_models.preferred_model({"openrouter_model_name": "x/y:free"}) == "x/y:free"
