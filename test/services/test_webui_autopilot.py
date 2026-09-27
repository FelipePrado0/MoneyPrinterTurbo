import re
from datetime import datetime

from streamlit.testing.v1 import AppTest

from app.services import autopilot, video_history


def _dashboard_app():
    from webui import autopilot_panel

    autopilot_panel.render_dashboard(lambda key: key)


def _voice_box(at):
    return next(w for w in at.selectbox if str(w.key).startswith("autopilot_voice_select_"))


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


def test_dashboard_shows_audience_corrections_as_plain_data():
    from app.services import audience_comments

    claim = "**tubarão** não é peixe [clique](https://evil.example)"
    video_history.set_state(
        audience_comments.STATE_KEY,
        {"requests": [], "corrections": [{"comment_id": "c1", "youtube_id": "yt1", "claim": claim}]},
    )
    at = AppTest.from_function(_dashboard_app, default_timeout=30).run()
    assert not at.exception
    assert [e.label for e in at.expander] == ["Autopilot Audience Corrections"]
    corrections = at.expander[0].dataframe[0].value
    assert corrections.iloc[0]["claim"] == claim
    assert corrections.iloc[0]["link"] == "https://www.youtube.com/watch?v=yt1&lc=c1"
    assert all(claim not in m.value for m in at.markdown)


def test_settings_form_saves_valid_values():
    # Edge voice: the form alone, without the Gemini options above it.
    autopilot.save_settings({"voice_name": "pt-BR-FranciscaNeural-Female"})
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
    # Edge voice: the form alone, without the Gemini options above it.
    autopilot.save_settings({"voice_name": "pt-BR-FranciscaNeural-Female"})
    at = AppTest.from_function(_settings_app, default_timeout=30).run()
    at.text_input[0].set_value("")
    at.button[0].click().run()
    assert not at.exception
    assert at.error and at.error[0].value.startswith("Autopilot Save Failed")
    assert autopilot.load_settings().video_language == "pt-BR"


def test_settings_form_picks_voice_from_list_and_keeps_unknown_saved_voice():
    autopilot.save_settings({"voice_name": "custom:minha-voz"})
    at = AppTest.from_function(_settings_app, default_timeout=30).run()
    assert not at.exception
    voice_box = _voice_box(at)
    assert voice_box.value == "custom:minha-voz"
    assert {"gemini:Kore-Firm", "pt-BR-FranciscaNeural-Female"} <= set(voice_box.options)

    voice_box.set_value("gemini:Kore-Firm").run()
    assert not at.exception
    assert autopilot.load_settings().voice_name == "gemini:Kore-Firm"


def _gemini_keys(at):
    """Gemini widget keys without the value hash (and key-row index) suffix."""
    widgets = [*at.text_input, *at.text_area, *at.multiselect, *at.selectbox]
    return {
        re.sub(r"_[0-9a-f]{8}(_\d+)?$", "", str(w.key))
        for w in widgets
        if str(w.key).startswith("autopilot_gemini_")
    }


def test_gemini_voice_shows_gemini_options_and_locks_speed():
    from app.config import config

    autopilot.save_settings({"voice_name": "pt-BR-FranciscaNeural-Female"})
    at = AppTest.from_function(_settings_app, default_timeout=30).run()
    assert _gemini_keys(at) == set()
    assert not at.slider[0].disabled

    _voice_box(at).set_value("gemini:Kore-Firm").run()
    assert not at.exception
    assert _gemini_keys(at) == {
        "autopilot_gemini_key",
        "autopilot_gemini_tts_style_input",
        "autopilot_gemini_tts_model_input",
        "autopilot_gemini_tts_fallback_voices_input",
        "autopilot_gemini_tts_last_resort_voice_input",
    }
    assert at.slider[0].disabled

    original = config.app.get("gemini_tts_style")
    try:
        style = next(w for w in at.text_area if str(w.key).startswith("autopilot_gemini_tts_style_input_"))
        style.set_value("Fale calmo:").run()
        assert config.app["gemini_tts_style"] == "Fale calmo:"
    finally:
        config.app["gemini_tts_style"] = original


def test_history_shows_narration_voice():
    video_history.record("t-voice", "Polvo", "autopilot")
    video_history.set_tts_voice("t-voice", "gemini:Puck")
    at = AppTest.from_function(_dashboard_app, default_timeout=30).run()
    assert not at.exception
    assert at.dataframe[-1].value.iloc[0]["voice"] == "gemini:Puck"
