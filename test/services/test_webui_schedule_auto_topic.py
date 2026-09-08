"""WebUI smoke tests for the "Tema automático" schedule mode.

The actual decision logic (cadence math, topic picking, repeat-avoidance)
is already covered by test_schedule_rules.py, test_trend_topic.py and
test_scheduler.py. This file only checks the UI wiring: the checkbox swaps
in the auto-topic form, and confirming it calls schedule_store.create_schedule
with the sentinel subject on every occurrence - it never runs a real trend
fetch, LLM call, or video generation.
"""

from datetime import time
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import schedule_store, trend_topic

ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


def _widget_by_key(elements, key):
    return next(
        (item for item in elements if str(getattr(item, "key", "")) == key), None
    )


@pytest.fixture
def schedule_app():
    with patch.object(config, "try_save_config", return_value=True):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.run()
        open_button = _widget_by_key(app.button, "open_schedule_dialog_button")
        assert open_button is not None, "schedule dialog entry button not found"
        open_button.click()
        app.run()
        yield app


def test_auto_topic_checkbox_swaps_in_the_auto_form(schedule_app):
    app = schedule_app
    checkbox = _widget_by_key(app.checkbox, "schedule_auto_topic_mode")
    assert checkbox is not None

    checkbox.set_value(True)
    app.run()

    assert not app.exception
    assert _widget_by_key(app.number_input, "schedule_auto_interval_hours") is not None

    # Re-fetch: the widget tree was rebuilt by the run above, and the old
    # reference no longer reflects into the next one.
    checkbox = _widget_by_key(app.checkbox, "schedule_auto_topic_mode")
    checkbox.set_value(False)
    app.run()

    assert not app.exception
    # Back in manual mode: its own distinctive widget must render again.
    assert _widget_by_key(app.date_input, "schedule_start_date") is not None


def test_confirm_creates_schedule_with_sentinel_subject_on_every_row(schedule_app):
    app = schedule_app
    checkbox = _widget_by_key(app.checkbox, "schedule_auto_topic_mode")
    checkbox.set_value(True)
    app.run()

    # Fixed at midnight so a 12h interval lands cleanly on 2 slots/day
    # (00:00, 12:00) instead of depending on whatever time the test runs at.
    start_time_input = _widget_by_key(app.time_input, "schedule_auto_start_time")
    start_time_input.set_value(time(0, 0))
    interval_input = _widget_by_key(
        app.number_input, "schedule_auto_interval_hours"
    )
    interval_input.set_value(12.0)
    day_count_input = _widget_by_key(app.number_input, "schedule_auto_day_count")
    day_count_input.set_value(2)
    app.run()

    with patch.object(schedule_store, "create_schedule") as mocked_create:
        confirm_button = _widget_by_key(app.button, "schedule_auto_confirm")
        assert confirm_button is not None
        confirm_button.click()
        app.run()

    assert not app.exception
    mocked_create.assert_called_once()
    _, kwargs = mocked_create.call_args
    occurrences = kwargs["occurrences"]
    assert len(occurrences) == 4  # 12h interval, 2 days -> 2 slots/day
    assert all(
        occurrence["video_subject"] == trend_topic.AUTO_TOPIC_SENTINEL
        for occurrence in occurrences
    )
