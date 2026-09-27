from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import autopilot

KEYS = ["key-a", "key-b", "key-c"]


def _voice_box(at):
    return next(w for w in at.selectbox if str(w.key).startswith("autopilot_voice_select_"))


def _settings_app():
    from webui import autopilot_panel

    autopilot_panel.render_settings_form(lambda key: key)


def _key_values(at):
    return [
        w.value
        for w in at.text_input
        if str(w.key).startswith("autopilot_gemini_key_")
    ]


def _run(test):
    app_cfg = dict(config.app, gemini_api_keys=list(KEYS))
    with patch.object(config, "app", app_cfg), patch.object(config, "save_config"):
        test(app_cfg)


def test_keys_survive_switching_voice_away_and_back():
    def test(app_cfg):
        autopilot.save_settings({"voice_name": "gemini:Kore-Firm"})
        at = AppTest.from_function(_settings_app, default_timeout=30).run()
        assert _key_values(at) == KEYS

        _voice_box(at).set_value("pt-BR-FranciscaNeural-Female").run()
        assert _key_values(at) == []
        _voice_box(at).set_value("gemini:Kore-Firm").run()
        assert _key_values(at) == KEYS
        at.run()
        assert _key_values(at) == KEYS
        assert app_cfg["gemini_api_keys"] == KEYS
        assert not at.exception

    _run(test)


def test_keys_can_be_added_and_removed_with_buttons():
    def test(app_cfg):
        autopilot.save_settings({"voice_name": "gemini:Kore-Firm"})
        at = AppTest.from_function(_settings_app, default_timeout=30).run()

        at.button(key="autopilot_gemini_key_add").click().run()
        assert _key_values(at) == [*KEYS, ""]
        new_field = [w for w in at.text_input if str(w.key).startswith("autopilot_gemini_key_")][-1]
        new_field.set_value("key-d").run()
        assert app_cfg["gemini_api_keys"] == [*KEYS, "key-d"]
        assert _key_values(at) == [*KEYS, "key-d"]

        at.button(key="autopilot_gemini_key_remove_0").click().run()
        assert app_cfg["gemini_api_keys"] == ["key-b", "key-c", "key-d"]
        assert _key_values(at) == ["key-b", "key-c", "key-d"]
        assert not at.exception

    _run(test)


def test_stale_panel_never_overwrites_keys_changed_elsewhere():
    def test(app_cfg):
        autopilot.save_settings({"voice_name": "gemini:Kore-Firm"})
        at = AppTest.from_function(_settings_app, default_timeout=30).run()
        # Another panel (or the dialog fragment) saves a 4th key meanwhile.
        app_cfg["gemini_api_keys"] = [*KEYS, "key-d"]

        stale = [w for w in at.text_input if str(w.key).startswith("autopilot_gemini_key_")]
        stale[0].set_value("key-z").run()
        assert app_cfg["gemini_api_keys"] == [*KEYS, "key-d"]
        assert _key_values(at) == [*KEYS, "key-d"]
        assert any("Gemini API Keys Changed Elsewhere" in e.value for e in at.warning)

        at.button(key="autopilot_gemini_key_remove_0").click().run()
        assert app_cfg["gemini_api_keys"] == ["key-b", "key-c", "key-d"]
        assert not at.exception

    _run(test)
