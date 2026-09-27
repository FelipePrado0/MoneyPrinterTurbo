"""Gemini TTS settings shared by the audio panel and the autopilot settings.

The config is the single source of truth. Every widget gets its value from
the config and a key that carries a hash of that value: when the config
changes (here or in the other panel) the widget is recreated with the new
value, and a widget that was hidden (non-Gemini voice) comes back with the
saved value instead of an empty state. Only a widget's own callback writes.
"""

import hashlib
import json

import streamlit as st

from app.config import config
from app.services import voice, voice_gemini

_STYLES = dict(voice_gemini.GEMINI_TTS_VOICES)


def _rev(value) -> str:
    return hashlib.sha1(json.dumps(value, ensure_ascii=False).encode()).hexdigest()[:8]


def _save(field: str, value) -> None:
    config.update_config_nonblocking(config.app, field, value)


def _saved_keys() -> list[str]:
    keys = config.app.get("gemini_api_keys") or []
    if isinstance(keys, str):
        keys = keys.split(",")
    keys = [key.strip() for key in keys if key and key.strip()]
    # The LLM key is only a default until the list is saved once.
    legacy = (config.app.get("gemini_api_key") or "").strip()
    return keys or ([legacy] if legacy else [])


def _render_keys(tr, prefix: str) -> None:
    """One masked field per key, in fallback order, with remove and add buttons."""
    keys = _saved_keys()
    extra_state = f"{prefix}gemini_key_extra"
    rows = [*keys, *[""] * st.session_state.get(extra_state, 0)] or [""]
    rev = _rev(keys)
    field_keys = [f"{prefix}gemini_key_{rev}_{index}" for index in range(len(rows))]

    def current_values() -> list[str]:
        return [
            str(st.session_state.get(field, value) or "").strip()
            for field, value in zip(field_keys, rows)
        ]

    stale_state = f"{prefix}gemini_key_stale"

    def changed_elsewhere() -> bool:
        # The settings dialog reruns alone, so the other panel can still show an
        # older list; saving from it would drop keys added meanwhile.
        if _rev(_saved_keys()) == rev:
            return False
        st.session_state[stale_state] = True
        return True

    def store(values: list[str]) -> None:
        if changed_elsewhere():
            return
        # Rows left empty stay on screen until filled or removed.
        st.session_state[extra_state] = sum(1 for value in values if not value)
        _save("gemini_api_keys", [value for value in values if value])

    def remove(index: int) -> None:
        values = current_values()
        del values[index]
        store(values)

    def add() -> None:
        st.session_state[extra_state] = st.session_state.get(extra_state, 0) + 1

    st.markdown(tr("Gemini API Key"))
    st.caption(tr("Gemini API Keys Help"))
    if st.session_state.pop(stale_state, False):
        st.warning(tr("Gemini API Keys Changed Elsewhere"))
    for index, (field, value) in enumerate(zip(field_keys, rows)):
        label = (
            tr("Gemini API Key Primary")
            if index == 0
            else tr("Gemini API Key Fallback").format(n=index + 1)
        )
        col_input, col_remove = st.columns([12, 1], vertical_alignment="bottom")
        col_input.text_input(
            label,
            value=value,
            type="password",
            key=field,
            on_change=lambda: store(current_values()),
        )
        col_remove.button(
            ":material/delete:",
            key=f"{prefix}gemini_key_remove_{index}",
            help=tr("Gemini API Key Remove"),
            on_click=remove,
            args=(index,),
        )
    st.button(
        tr("Gemini API Key Add"),
        icon=":material/add:",
        key=f"{prefix}gemini_key_add",
        on_click=add,
    )


def render(tr, prefix: str = "") -> None:
    """Keys (fallback order), speaking instruction, model and fallback voices."""
    _render_keys(tr, prefix)

    style = config.app.get("gemini_tts_style", "")
    style_key = f"{prefix}gemini_tts_style_input_{_rev(style)}"
    st.text_area(
        tr("Gemini TTS Style"),
        value=style,
        height=100,
        help=tr("Gemini TTS Style Help"),
        key=style_key,
        on_change=lambda: _save("gemini_tts_style", st.session_state[style_key].strip()),
    )

    model = config.app.get("gemini_tts_model") or voice_gemini.DEFAULT_TTS_MODEL
    model_key = f"{prefix}gemini_tts_model_input_{_rev(model)}"
    st.text_input(
        tr("Gemini TTS Model"),
        value=model,
        help=tr("Gemini TTS Model Help"),
        key=model_key,
        on_change=lambda: _save(
            "gemini_tts_model",
            st.session_state[model_key].strip() or voice_gemini.DEFAULT_TTS_MODEL,
        ),
    )

    fallbacks = [
        name
        for name in config.app.get(
            "gemini_tts_fallback_voices", list(voice_gemini.DEFAULT_FALLBACK_VOICES)
        )
        if name in _STYLES
    ]
    fallbacks_key = f"{prefix}gemini_tts_fallback_voices_input_{_rev(fallbacks)}"
    st.multiselect(
        tr("Gemini TTS Fallback Voices"),
        options=list(_STYLES),
        default=fallbacks,
        format_func=lambda name: f"{name} ({_STYLES[name]})",
        help=tr("Gemini TTS Fallback Voices Help"),
        key=fallbacks_key,
        on_change=lambda: _save(
            "gemini_tts_fallback_voices", list(st.session_state[fallbacks_key])
        ),
    )

    # "No last-resort voice" is stored as "" but shown by its own label, used as
    # the option value so the value always equals the text on screen.
    none_label = tr("Gemini TTS Last Resort None")
    last_resort = config.app.get("gemini_tts_last_resort_voice", "") or none_label
    options = [none_label, *voice.get_all_azure_voices()]
    if last_resort not in options:
        options.insert(1, last_resort)
    last_resort_key = f"{prefix}gemini_tts_last_resort_voice_input_{_rev(last_resort)}"
    st.selectbox(
        tr("Gemini TTS Last Resort Voice"),
        options=options,
        index=options.index(last_resort),
        help=tr("Gemini TTS Last Resort Voice Help"),
        key=last_resort_key,
        on_change=lambda: _save(
            "gemini_tts_last_resort_voice",
            ""
            if st.session_state[last_resort_key] == none_label
            else st.session_state[last_resort_key],
        ),
    )
