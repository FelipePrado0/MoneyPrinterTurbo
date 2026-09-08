"""WebUI tests for the "Clip From Video" section (upload -> STT -> <=60s
vertical clip). Real STT/ffmpeg/video generation never runs here:
``clip_service.save_clip_source`` and ``webui_task.submit_clip_generation``
are mocked, matching the rest of this suite's approach to controller/UI
tests and keeping these tests fast and free of real video generation.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import clip as clip_service
from app.services import webui_task

ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


def _widget_by_key(elements, key):
    return next(
        (item for item in elements if str(getattr(item, "key", "")) == key), None
    )


@pytest.fixture
def clip_app(tmp_path, monkeypatch):
    tasks_dir = tmp_path / "storage" / "tasks"
    tasks_dir.mkdir(parents=True)

    with patch.object(config, "try_save_config", return_value=True):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.run()
        yield app


def _upload_signature(app, filename="episode.mp4", size=1234):
    file_uploader = _widget_by_key(app.get("file_uploader"), "clip_source_uploader")
    assert file_uploader is not None, "clip source uploader not found"
    file_uploader.set_value((filename, b"fake mp4 bytes", "video/mp4"))
    return file_uploader


class TestClipFromVideoSection:
    def test_section_renders_uploader(self, clip_app):
        app = clip_app
        uploader = _widget_by_key(app.get("file_uploader"), "clip_source_uploader")
        assert uploader is not None
        assert not app.exception

    def test_upload_shows_start_time_and_generate_controls(self, clip_app, tmp_path):
        app = clip_app
        with patch.object(
            clip_service,
            "save_clip_source",
            return_value=("stub-upload.mp4", 90.0, True),
        ):
            _upload_signature(app)
            app.run()

        assert not app.exception
        start_time_input = _widget_by_key(
            app.get("number_input"), "clip_start_time"
        )
        assert start_time_input is not None
        assert start_time_input.max == pytest.approx(
            90.0 - clip_service.MIN_REMAINING_SECONDS
        )
        generate_button = _widget_by_key(app.button, "clip_generate_button")
        assert generate_button is not None

    def test_invalid_upload_shows_error_and_no_controls(self, clip_app):
        app = clip_app
        with patch.object(
            clip_service,
            "save_clip_source",
            side_effect=clip_service.ClipValidationError(
                "uploaded video has no audio track; speech-to-text needs audio"
            ),
        ):
            _upload_signature(app)
            app.run()

        assert not app.exception
        assert any(
            "speech-to-text needs audio" in error.value for error in app.error
        )
        assert _widget_by_key(app.button, "clip_generate_button") is None

    def test_generate_click_submits_task_with_chosen_start_time(self, clip_app):
        app = clip_app
        with patch.object(
            clip_service,
            "save_clip_source",
            return_value=("stub-upload.mp4", 90.0, True),
        ):
            _upload_signature(app)
            app.run()

        start_time_input = _widget_by_key(app.get("number_input"), "clip_start_time")
        start_time_input.set_value(20.0)
        app.run()

        with patch.object(
            webui_task, "submit_clip_generation"
        ) as mocked_submit, patch.object(
            clip_service, "validate_start_time"
        ) as mocked_validate:
            generate_button = _widget_by_key(app.button, "clip_generate_button")
            generate_button.click()
            app.run()

        assert not app.exception
        mocked_validate.assert_called_once()
        mocked_submit.assert_called_once()
        _, submit_kwargs = mocked_submit.call_args
        assert submit_kwargs["upload_id"] == "stub-upload.mp4"
        assert submit_kwargs["start_time"] == 20.0

    def test_generate_click_blocked_by_start_time_validation_error(self, clip_app):
        app = clip_app
        with patch.object(
            clip_service,
            "save_clip_source",
            return_value=("stub-upload.mp4", 90.0, True),
        ):
            _upload_signature(app)
            app.run()

        with patch.object(
            webui_task, "submit_clip_generation"
        ) as mocked_submit, patch.object(
            clip_service,
            "validate_start_time",
            side_effect=clip_service.ClipValidationError(
                "start_time leaves less than 3.0s of video remaining"
            ),
        ):
            generate_button = _widget_by_key(app.button, "clip_generate_button")
            generate_button.click()
            app.run()

        assert not app.exception
        mocked_submit.assert_not_called()
        assert any(
            "remaining" in error.value for error in app.error
        )
