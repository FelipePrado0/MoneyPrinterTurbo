"""API tests for the clip-from-video controller.

Real STT/ffmpeg work never runs here: ``clip_service.save_clip_source`` and
the task scheduler are mocked, matching how the rest of this suite tests
controllers (mock the service layer, assert routing/response/status-code
behavior) and keeping these tests fast and free of real video generation.
"""

import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app import asgi
from app.config import config
from app.controllers.v1 import clip as clip_controller
from app.models import const
from app.services import clip as clip_service
from app.services import state as sm


class TestClipControllerHTTP(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["api_key"] = ""
        self.client = TestClient(asgi.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_upload_clip_source_returns_upload_id_duration_and_audio_flag(self):
        with mock.patch.object(
            clip_service,
            "save_clip_source",
            return_value=("abc123.mp4", 812.5, True),
        ) as mocked_save:
            response = self.client.post(
                "/api/v1/clip_uploads",
                files={"file": ("episode.mp4", b"fake video bytes", "video/mp4")},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["upload_id"], "abc123.mp4")
        self.assertEqual(data["duration"], 812.5)
        self.assertTrue(data["has_audio"])
        mocked_save.assert_called_once()

    def test_upload_clip_source_rejects_invalid_video_with_400(self):
        with mock.patch.object(
            clip_service,
            "save_clip_source",
            side_effect=clip_service.ClipValidationError("uploaded file is empty"),
        ):
            response = self.client.post(
                "/api/v1/clip_uploads",
                files={"file": ("episode.mp4", b"", "video/mp4")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("uploaded file is empty", response.json()["message"])

    def test_upload_clip_source_maps_unexpected_error_to_500(self):
        with mock.patch.object(
            clip_service,
            "save_clip_source",
            side_effect=RuntimeError("ffmpeg exploded"),
        ):
            response = self.client.post(
                "/api/v1/clip_uploads",
                files={"file": ("episode.mp4", b"data", "video/mp4")},
            )

        self.assertEqual(response.status_code, 500)
        # Internal exception details must not leak to the client.
        self.assertNotIn("ffmpeg exploded", response.text)

    def test_create_clip_task_schedules_generate_clip_and_returns_task_id(self):
        with mock.patch.object(clip_controller.task_manager, "add_task") as mocked_add:
            response = self.client.post(
                "/api/v1/clip_tasks",
                json={
                    "upload_id": "abc123.mp4",
                    "start_time": 12.0,
                    "subtitle_position": "bottom",
                    "subtitle_display_mode": "sentence",
                },
            )

        self.assertEqual(response.status_code, 200)
        task_id = response.json()["data"]["task_id"]
        self.assertTrue(task_id)

        mocked_add.assert_called_once()
        call_args, call_kwargs = mocked_add.call_args
        self.assertEqual(call_args[0], clip_service.generate_clip)
        self.assertEqual(call_kwargs["task_id"], task_id)
        self.assertEqual(call_kwargs["upload_id"], "abc123.mp4")
        self.assertEqual(call_kwargs["start_time"], 12.0)

        # The task record must exist and be tagged "clip" even though the
        # (mocked) worker never actually ran.
        task = sm.state.get_task(task_id)
        self.assertEqual(task["kind"], "clip")

    def test_create_clip_task_rejects_missing_upload_id(self):
        response = self.client.post(
            "/api/v1/clip_tasks",
            json={"start_time": 0.0},
        )
        # This app maps request validation errors to 400, not FastAPI's
        # default 422 (see app.asgi.validation_exception_handler).
        self.assertEqual(response.status_code, 400)

    def test_get_clip_task_returns_status_for_known_clip_task(self):
        sm.state.update_task(
            "clip-task-1",
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            kind="clip",
            video_file="/tasks/clip-task-1/final.mp4",
            hard_cut_fallback=False,
        )
        try:
            response = self.client.get("/api/v1/clip_tasks/clip-task-1")
        finally:
            sm.state.delete_task("clip-task-1")

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["state"], const.TASK_STATE_COMPLETE)
        self.assertFalse(data["hard_cut_fallback"])

    def test_get_clip_task_404_for_unknown_task(self):
        response = self.client.get("/api/v1/clip_tasks/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_get_clip_task_404_for_non_clip_task(self):
        # A regular video-generation task must not be exposed through the
        # clip status endpoint even if the ID happens to exist.
        sm.state.update_task("video-task-1", state=const.TASK_STATE_PROCESSING)
        try:
            response = self.client.get("/api/v1/clip_tasks/video-task-1")
        finally:
            sm.state.delete_task("video-task-1")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
