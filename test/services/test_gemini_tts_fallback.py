import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loguru import logger
from pydub import AudioSegment

from app.services import task as task_service
from app.services import video_history
from app.services import voice as vs

KEYS = ["key-handplays", "key-unaerp", "key-felipegreck"]


def _pcm(duration_ms=1200):
    return (
        AudioSegment.silent(duration=duration_ms)
        .set_frame_rate(24000)
        .set_channels(1)
        .set_sample_width(2)
        .raw_data
    )


class _Response:
    def __init__(self, data):
        part = type("Part", (), {"inline_data": type("D", (), {"data": data})()})()
        content = type("Content", (), {"parts": [part]})()
        self.candidates = [type("Candidate", (), {"content": content})()]


def _fake_client(calls, succeed_on):
    """Client that records (voice, key) and only answers for ``succeed_on``."""

    class _Models:
        def __init__(self, key):
            self.key = key

        def generate_content(self, **kwargs):
            voice = kwargs["config"].speech_config.voice_config.prebuilt_voice_config.voice_name
            calls.append((voice, self.key, kwargs["model"], kwargs["contents"]))
            if (voice, self.key) not in succeed_on:
                raise RuntimeError(f"429 RESOURCE_EXHAUSTED for {self.key}")
            return _Response(_pcm())

    class _Client:
        def __init__(self, api_key, **kwargs):
            self.models = _Models(api_key)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Client


class GeminiTtsFallbackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gemini-fallback-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.voice_file = str(self.tmp / "audio.mp3")
        self.app_cfg = dict(
            vs.config.app,
            gemini_api_keys=list(KEYS),
            gemini_api_key="",
            gemini_tts_model="gemini-3.1-flash-tts-preview",
            gemini_tts_style="Narre como documentário:",
            gemini_tts_fallback_voices=["Puck", "Zephyr"],
        )

    def _run(self, succeed_on, app_cfg=None):
        calls = []
        with patch("google.genai.Client", _fake_client(calls, succeed_on)), patch.object(
            vs.config, "app", app_cfg or self.app_cfg
        ):
            sub_maker = vs.gemini_tts("O polvo tem três corações.", "Kore", 1.0, self.voice_file)
        return sub_maker, calls

    def test_uses_keys_in_configured_order_until_one_works(self):
        sub_maker, calls = self._run({("Kore", KEYS[1])})

        self.assertEqual([(v, k) for v, k, _, _ in calls], [("Kore", KEYS[0]), ("Kore", KEYS[1])])
        self.assertIsNotNone(sub_maker)
        self.assertEqual(sub_maker.tts_voice, "gemini:Kore")
        self.assertTrue(sub_maker.needs_alignment)
        self.assertTrue(Path(self.voice_file).is_file())

    def test_falls_back_to_next_voice_after_every_key_fails(self):
        sub_maker, calls = self._run({("Zephyr", KEYS[2])})

        expected = [(v, k) for v in ("Kore", "Puck", "Zephyr") for k in KEYS]
        self.assertEqual([(v, k) for v, k, _, _ in calls], expected)
        self.assertEqual(sub_maker.tts_voice, "gemini:Zephyr")

    def test_returns_none_when_all_fail_and_never_logs_keys(self):
        messages = []
        sink = logger.add(lambda m: messages.append(str(m)), level="DEBUG")
        self.addCleanup(logger.remove, sink)

        sub_maker, calls = self._run(set())

        self.assertIsNone(sub_maker)
        self.assertEqual(len(calls), 9)
        log = "".join(messages)
        self.assertIn("key#1", log)
        for key in KEYS:
            self.assertNotIn(key, log)

    def test_sends_style_and_configured_model(self):
        _, calls = self._run({("Kore", KEYS[0])})

        _, _, model, contents = calls[0]
        self.assertEqual(model, "gemini-3.1-flash-tts-preview")
        self.assertEqual(contents, "Narre como documentário: O polvo tem três corações.")

    def test_legacy_single_key_still_works(self):
        cfg = dict(self.app_cfg, gemini_api_keys=[], gemini_api_key="legacy-key")
        sub_maker, calls = self._run({("Kore", "legacy-key")}, cfg)

        self.assertIsNotNone(sub_maker)
        self.assertEqual(calls[0][1], "legacy-key")


class _Params:
    voice_name = "gemini:Kore-Firm"
    voice_rate = 0.9
    custom_audio_file = None
    subtitle_enabled = True


class LastResortVoiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gemini-last-resort-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        video_history.record("task-lr", "Polvo", "autopilot")

    def _generate(self, tts_side_effect):
        def fake_tts(text, voice_name, voice_rate, voice_file, voice_volume=1.0):
            result = tts_side_effect(voice_name)
            if result is not None:
                Path(voice_file).write_bytes(b"x")
            return result

        with patch.object(vs, "tts", side_effect=fake_tts) as tts_mock, patch.object(
            vs, "get_audio_duration", return_value=3.0
        ), patch(
            "app.utils.utils.task_dir", lambda tid="": str(self.tmp)
        ), patch.object(
            task_service.config,
            "app",
            dict(task_service.config.app, gemini_tts_last_resort_voice="pt-BR-FranciscaNeural-Female"),
        ):
            result = task_service.generate_audio("task-lr", _Params(), "O polvo tem três corações.")
        return result, tts_mock

    def test_uses_edge_last_resort_when_every_gemini_voice_fails(self):
        edge = vs.ensure_legacy_submaker_fields(vs.SubMaker())
        _, _, sub_maker = self._generate(lambda v: None if v.startswith("gemini:") else edge)[0]

        self.assertIs(sub_maker, edge)
        self.assertEqual(sub_maker.tts_voice, "pt-BR-FranciscaNeural-Female")
        video = video_history.list_videos()[0][0]
        self.assertEqual(video["tts_voice"], "pt-BR-FranciscaNeural-Female")

    def test_records_gemini_voice_actually_used(self):
        gemini = vs.ensure_legacy_submaker_fields(vs.SubMaker())
        gemini.tts_voice = "gemini:Puck"
        (_, _, _), tts_mock = self._generate(lambda v: gemini)

        self.assertEqual(tts_mock.call_count, 1)
        self.assertEqual(video_history.list_videos()[0][0]["tts_voice"], "gemini:Puck")


class TtsFallbackWarningTest(unittest.TestCase):
    def test_no_warning_when_requested_gemini_voice_spoke(self):
        self.assertIsNone(task_service.tts_fallback_warning("gemini:Kore-Firm", "gemini:Kore"))

    def test_warns_when_another_gemini_voice_spoke(self):
        self.assertEqual(
            task_service.tts_fallback_warning("gemini:Kore-Firm", "gemini:Puck"),
            {"code": "tts_voice_fallback", "requested": "gemini:Kore-Firm", "used": "gemini:Puck"},
        )

    def test_warns_when_last_resort_voice_spoke(self):
        warning = task_service.tts_fallback_warning(
            "gemini:Kore-Firm", "pt-BR-FranciscaNeural-Female"
        )
        self.assertEqual(warning["used"], "pt-BR-FranciscaNeural-Female")

    def test_no_warning_for_non_gemini_voices(self):
        self.assertIsNone(
            task_service.tts_fallback_warning("pt-BR-FranciscaNeural-Female", "pt-BR-FranciscaNeural-Female")
        )
        self.assertIsNone(task_service.tts_fallback_warning("gemini:Kore-Firm", ""))


class ReusedPreviewVoiceTest(unittest.TestCase):
    def test_reused_preview_records_the_voice_that_spoke(self):
        tmp = Path(tempfile.mkdtemp(prefix="gemini-preview-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        video_history.record("task-preview", "Polvo", "autopilot")
        sub_maker = vs.ensure_legacy_submaker_fields(vs.SubMaker())
        sub_maker.tts_voice = "gemini:Zephyr"

        with patch.object(
            task_service, "_resolve_reusable_voice_preview", return_value=("a.mp3", 3, sub_maker)
        ), patch("app.utils.utils.task_dir", lambda tid="": str(tmp)):
            result = task_service.generate_audio("task-preview", _Params(), "Polvo.", {"x": 1})

        self.assertIs(result[2], sub_maker)
        self.assertEqual(video_history.list_videos()[0][0]["tts_voice"], "gemini:Zephyr")
