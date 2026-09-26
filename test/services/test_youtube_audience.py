"""YouTube audience ("made for kids") declaration on the native upload path.

Upstream v1.3.7 added this for Upload-Post's YouTube target; this fork
publishes YouTube through the official Data API instead (Upload-Post only
serves TikTok/Instagram), so the same guarantees are enforced there:
strict boolean values, and the value captured when the upload is queued.
"""

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config
from app.models.schema import VideoParams
from app.services import task, task_cross_post, youtube_upload
from app.services.state import MemoryState

WEBUI_MAIN = Path(__file__).resolve().parents[2] / "webui" / "Main.py"


@pytest.fixture
def service(tmp_path):
    video = tmp_path / "final.mp4"
    video.write_bytes(b"video")
    svc = youtube_upload.YouTubeUploadService()
    with patch.object(type(svc), "is_configured", return_value=True):
        yield svc, str(video)


def _captured_body(svc, video, **kwargs):
    client = MagicMock()
    with (
        patch.object(youtube_upload, "_load_google_modules", return_value=MagicMock()),
        patch.object(svc, "_build_client", return_value=client),
        patch.object(svc, "_execute_resumable_upload", return_value={"id": "vid1"}),
    ):
        result = svc.upload_video(video_path=video, title="t", **kwargs)
    assert result["success"], result
    return client.videos().insert.call_args.kwargs["body"]


@pytest.mark.parametrize("configured, explicit, expected", [
    (False, None, False),
    (True, None, True),
    (False, True, True),
    (True, False, False),
])
def test_audience_payload_and_snapshot_override(service, configured, explicit, expected):
    svc, video = service
    with patch.dict(config.app, {"youtube_made_for_kids": configured}):
        body = _captured_body(svc, video, made_for_kids=explicit)
    assert body["status"]["selfDeclaredMadeForKids"] is expected


@pytest.mark.parametrize("invalid", ["false", "true", "1", 2])
def test_invalid_audience_never_uploads(service, invalid):
    svc, video = service
    with (
        patch.dict(config.app, {"youtube_made_for_kids": invalid}),
        patch.object(youtube_upload, "_load_google_modules") as load_modules,
    ):
        result = svc.upload_video(video_path=video, title="t")
    assert result["success"] is False
    assert "boolean" in result["error"]
    load_modules.assert_not_called()


@pytest.mark.parametrize("selected", [True, False])
def test_queued_audience_survives_config_change(selected):
    state = MemoryState()
    state.update_task("audience-snapshot", state=task.const.TASK_STATE_COMPLETE)
    future = Future()
    with (
        patch.object(task.sm, "state", state),
        patch.object(task._cross_post_executor, "submit", return_value=future) as submit,
        patch.object(
            type(youtube_upload.youtube_upload_service),
            "made_for_kids",
            new_callable=PropertyMock,
            return_value=selected,
        ),
    ):
        task._schedule_cross_post(
            task_id="audience-snapshot",
            video_paths=["final.mp4"],
            params=VideoParams(video_subject="Coffee"),
            video_script="script",
            platforms=[],
            publish_youtube=True,
            youtube_privacy_status="public",
            youtube_made_for_kids=selected,
        )
    queued_args = submit.call_args.args
    with (
        patch.dict(config.app, {"youtube_made_for_kids": not selected}),
        patch.object(task.sm, "state", state),
        patch.object(task.llm, "generate_social_metadata", return_value={}),
        patch.object(
            task.youtube_upload, "publish_video", return_value={"success": True, "video_id": "v"}
        ) as publish,
        patch.object(task_cross_post.webhook_notifier, "notify_video_published"),
    ):
        queued_args[0](*queued_args[1:])
    assert publish.call_args.kwargs["made_for_kids"] is selected


def _widget_by_key(widgets, key):
    return next(widget for widget in widgets if widget.key == key)


@pytest.mark.parametrize("saved", [False, True, "false"])
def test_webui_audience_keeps_invalid_value_until_user_chooses(saved):
    values = dict(config.app, youtube_made_for_kids=saved)
    with (
        patch.object(config, "app", values),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60).run()
        app.session_state["settings_dialog_open"] = True
        app.run()
        assert not app.exception
        selector = _widget_by_key(app.selectbox, "youtube_made_for_kids_selectbox")
        assert selector.value is (saved if isinstance(saved, bool) else None)
        assert values["youtube_made_for_kids"] == saved
        for value in (True, False):
            _widget_by_key(app.selectbox, "youtube_made_for_kids_selectbox").set_value(value)
            app.run()
            assert not app.exception
            assert values["youtube_made_for_kids"] is value
