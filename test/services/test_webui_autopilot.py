from datetime import datetime

from streamlit.testing.v1 import AppTest

from app.services import autopilot, video_history


def _dashboard_app():
    from webui import autopilot_panel

    autopilot_panel.render_dashboard(lambda key: key)


def _settings_app():
    from webui import autopilot_panel

    autopilot_panel.render_settings_form(lambda key: key)


def _captions(at):
    return [caption.value for caption in at.caption]


def test_dashboard_empty_state_teaches_next_step():
    at = AppTest.from_function(_dashboard_app, default_timeout=30).run()
    assert not at.exception
    captions = _captions(at)
    assert "Autopilot Today Empty Disabled" in captions
    assert "Autopilot History Empty" in captions
    pause = at.button(key="autopilot_pause")
    assert pause.disabled


def test_dashboard_lists_today_and_history_and_pauses():
    now = datetime.now().replace(hour=0, minute=1)
    autopilot.save_settings(
        {"enabled": True, "start_time": "23:50", "videos_per_day": 1}, now=now
    )
    video_history.record("t1", "Polvo tem 3 corações", "autopilot", fact_key="polvo:3")
    video_history.mark_published("t1", "yt1", "https://www.youtube.com/watch?v=yt1")

    at = AppTest.from_function(_dashboard_app, default_timeout=30).run()
    assert not at.exception
    assert len(at.dataframe) == 2
    history = at.dataframe[1].value
    assert history.iloc[0]["subject"] == "Polvo tem 3 corações"

    at.button(key="autopilot_pause").click().run()
    assert not at.exception
    assert autopilot.is_paused()
    assert [toast.value for toast in at.toast] == ["Autopilot Paused Toast"]
    assert at.button(key="autopilot_resume")


def test_settings_form_saves_valid_values():
    at = AppTest.from_function(_settings_app, default_timeout=30).run()
    assert not at.exception
    at.toggle[0].set_value(True)
    at.number_input[0].set_value(2)
    at.text_area[0].set_value("astronomia e espaço")
    at.button[0].click().run()
    assert not at.exception
    settings = autopilot.load_settings()
    assert settings.enabled and settings.videos_per_day == 2
    assert settings.niche == "astronomia e espaço"
    assert [s.value for s in at.success] == ["Autopilot Saved"]


def test_settings_form_reports_invalid_values_without_saving():
    at = AppTest.from_function(_settings_app, default_timeout=30).run()
    at.text_input[0].set_value("")
    at.button[0].click().run()
    assert not at.exception
    assert at.error and at.error[0].value.startswith("Autopilot Save Failed")
    assert autopilot.load_settings().video_language == "pt-BR"
