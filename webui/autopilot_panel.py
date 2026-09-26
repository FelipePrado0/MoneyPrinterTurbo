"""WebUI for the channel autopilot: dashboard, settings form, LLM fallback.

Native Streamlit components only (dataframe column_config, badge, form):
they keep their alignment across Streamlit upgrades, unlike CSS aimed at
internal class names. Every text goes through the caller's ``tr``.
"""

from datetime import datetime, time

import streamlit as st

from app.services import autopilot, llm_free_models, schedule_store, video_history

HISTORY_LIMIT = 100

_SLOT_STATUS_KEYS = {
    schedule_store.STATUS_PENDING: "Autopilot Slot Pending",
    schedule_store.STATUS_DISPATCHED: "Autopilot Slot Running",
    schedule_store.STATUS_FAILED: "Autopilot Slot Failed",
}
_VIDEO_STATUS_KEYS = {
    video_history.STATUS_GENERATING: "Autopilot Video Generating",
    video_history.STATUS_GENERATED: "Autopilot Video Generated",
    video_history.STATUS_PUBLISHED: "Autopilot Video Published",
    video_history.STATUS_FAILED: "Autopilot Video Failed",
    video_history.STATUS_REJECTED: "Autopilot Video Rejected",
}
_MODEL_STATE_KEYS = {
    "ok": "Autopilot Model Ok",
    "cooldown": "Autopilot Model Cooldown",
    "unavailable": "Autopilot Model Unavailable",
}


def _dataframe(rows: list[dict], column_config: dict) -> None:
    """Hide columns with no value in any row (e.g. metrics before the first
    collection): a column of blanks reads as broken data, not as "none yet"."""
    filled = {key for row in rows for key, value in row.items() if value not in (None, "")}
    st.dataframe(
        [{key: value for key, value in row.items() if key in filled} for row in rows],
        hide_index=True,
        width="stretch",
        column_config={key: cfg for key, cfg in column_config.items() if key in filled},
    )


def _format_time(value) -> str:
    if not value:
        return "-"
    if isinstance(value, (int, float)):
        value = datetime.fromtimestamp(value)
    elif isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime("%d/%m %H:%M")


# --- dashboard ----------------------------------------------------------------


def _render_status_bar(tr, status: dict) -> None:
    settings = status["settings"]
    if not settings["enabled"]:
        label, color, icon = tr("Autopilot Off"), "gray", ":material/power_settings_new:"
    elif status["paused"]:
        label, color, icon = tr("Autopilot Paused"), "orange", ":material/pause_circle:"
    else:
        label, color, icon = tr("Autopilot Running"), "green", ":material/autoplay:"

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(label, color=color, icon=icon)
        st.badge(
            tr("Autopilot Uploads Today").format(
                used=status["uploads_today"], quota=settings["daily_upload_quota"]
            ),
            color="blue",
            icon=":material/upload:",
        )
        next_run = (
            _format_time(status["next_run"]) if status["next_run"] else tr("Autopilot None")
        )
        st.badge(
            tr("Autopilot Next Run").format(when=next_run),
            color="violet",
            icon=":material/schedule:",
        )


# Callbacks must not display elements: during a fragment rerun Streamlit would
# paint them over the top of the app and the frontend crashes. They leave the
# toast for the next render instead.
def _pause() -> None:
    autopilot.pause()
    st.session_state["autopilot_toast"] = ("Autopilot Paused Toast", ":material/pause:")


def _resume() -> None:
    autopilot.resume()
    st.session_state["autopilot_toast"] = ("Autopilot Resumed", ":material/play_arrow:")


def _open_settings() -> None:
    st.session_state["autopilot_dialog_open"] = False
    st.session_state["settings_dialog_open"] = True
    st.session_state["settings_dialog_target_tab"] = "autopilot"


def _render_actions(tr, status: dict) -> None:
    # on_click runs before the next render, so the status above is already
    # fresh without an explicit rerun (and a double click is a no-op).
    enabled = status["settings"]["enabled"]
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if status["paused"]:
            st.button(
                tr("Autopilot Resume"),
                key="autopilot_resume",
                icon=":material/play_arrow:",
                type="primary",
                disabled=not enabled,
                on_click=_resume,
            )
        else:
            st.button(
                tr("Autopilot Pause"),
                key="autopilot_pause",
                icon=":material/pause:",
                disabled=not enabled,
                help=tr("Autopilot Pause Help"),
                on_click=_pause,
            )
        if st.button(
            tr("Autopilot Configure"),
            key="autopilot_configure",
            icon=":material/tune:",
            on_click=_open_settings,
        ):
            st.rerun(scope="app")


def _render_today(tr, status: dict) -> None:
    st.markdown(f"**{tr('Autopilot Today')}**")
    slots = status["today"]
    if not slots:
        if not status["settings"]["enabled"]:
            st.caption(tr("Autopilot Today Empty Disabled"))
        elif status["paused"]:
            st.caption(tr("Autopilot Today Empty Paused"))
        else:
            st.caption(tr("Autopilot Today Empty"))
        return
    _dataframe(
        [
            {
                "when": _format_time(slot["generate_at"]),
                "subject": (
                    tr("Autopilot Topic Pending")
                    if slot["subject"] == autopilot.AUTO_TOPIC_SENTINEL
                    else slot["subject"]
                ),
                "status": tr(_SLOT_STATUS_KEYS.get(slot["status"], slot["status"])),
                "attempt": slot["attempt"],
                "error": slot["error"] or "",
            }
            for slot in slots
        ],
        {
            "when": st.column_config.TextColumn(tr("Autopilot Col When"), width="small"),
            "subject": st.column_config.TextColumn(tr("Autopilot Col Subject"), width="large"),
            "status": st.column_config.TextColumn(tr("Autopilot Col Status")),
            "attempt": st.column_config.NumberColumn(tr("Autopilot Col Attempt"), width="small"),
            "error": st.column_config.TextColumn(tr("Autopilot Col Error"), width="medium"),
        },
    )


def _render_history(tr) -> None:
    header_col, filter_col = st.columns([3, 1], vertical_alignment="bottom")
    header_col.markdown(f"**{tr('Autopilot History')}**")
    status_options = ["all", *video_history.STATUSES]
    selected = filter_col.selectbox(
        tr("Autopilot Filter Status"),
        options=status_options,
        format_func=lambda s: tr("Autopilot All") if s == "all" else tr(_VIDEO_STATUS_KEYS[s]),
        key="autopilot_history_filter",
        label_visibility="collapsed",
    )
    rows, total = video_history.list_videos(
        limit=HISTORY_LIMIT, status=None if selected == "all" else selected
    )
    if not rows:
        st.caption(tr("Autopilot History Empty"))
        return
    _dataframe(
        [
            {
                "created": _format_time(row["created_at"]),
                "subject": row["subject"],
                "status": tr(_VIDEO_STATUS_KEYS.get(row["status"], row["status"])),
                "views": row["views"],
                "retention": row["avg_view_percentage"],
                "link": row["youtube_url"],
                "model": row["llm_model"] or "",
                "error": row["error"] or "",
            }
            for row in rows
        ],
        {
            "created": st.column_config.TextColumn(tr("Autopilot Col When"), width="small"),
            "subject": st.column_config.TextColumn(tr("Autopilot Col Subject"), width="large"),
            "status": st.column_config.TextColumn(tr("Autopilot Col Status")),
            "views": st.column_config.NumberColumn(tr("Autopilot Col Views"), format="%d", width="small"),
            "retention": st.column_config.NumberColumn(
                tr("Autopilot Col Retention"), format="%.0f%%", width="small"
            ),
            "link": st.column_config.LinkColumn(
                tr("Autopilot Col Link"), display_text=tr("Autopilot Open"), width="small"
            ),
            "model": st.column_config.TextColumn(tr("Autopilot Col Model"), width="medium"),
            "error": st.column_config.TextColumn(tr("Autopilot Col Error"), width="medium"),
        },
    )
    if total > len(rows):
        st.caption(tr("Autopilot History Truncated").format(shown=len(rows), total=total))


@st.fragment(run_every="30s")
def render_dashboard(tr) -> None:
    toast = st.session_state.pop("autopilot_toast", None)
    if toast:
        st.toast(tr(toast[0]), icon=toast[1])
    try:
        status = autopilot.status()
    except Exception as exc:
        st.error(tr("Autopilot Load Failed").format(error=exc))
        return

    top_left, top_right = st.columns([3, 2], vertical_alignment="center")
    with top_left:
        _render_status_bar(tr, status)
    with top_right:
        _render_actions(tr, status)

    if "needs_reauthorization" in status["metrics_problems"]:
        st.warning(tr("Autopilot Metrics Reauthorize"), icon=":material/key:")
    elif status["metrics_problems"]:
        st.warning(
            tr("Autopilot Metrics Problem").format(error="; ".join(status["metrics_problems"])),
            icon=":material/warning:",
        )

    _render_today(tr, status)
    try:
        _render_history(tr)
    except Exception as exc:
        st.error(tr("Autopilot Load Failed").format(error=exc))


# --- settings form ------------------------------------------------------------


def render_settings_form(tr) -> None:
    try:
        current = autopilot.load_settings()
    except Exception as exc:
        st.error(tr("Autopilot Load Failed").format(error=exc))
        return

    st.caption(tr("Autopilot Settings Intro"))
    with st.form("autopilot_settings_form", border=False):
        enabled = st.toggle(tr("Autopilot Enable"), value=current.enabled)

        col_count, col_start, col_interval = st.columns(3)
        videos_per_day = col_count.number_input(
            tr("Autopilot Videos Per Day"), min_value=1, max_value=6, value=current.videos_per_day, step=1
        )
        hour, minute = (int(part) for part in current.start_time.split(":"))
        start_time = col_start.time_input(
            tr("Autopilot Start Time"), value=time(hour, minute), step=300
        )
        interval_minutes = col_interval.number_input(
            tr("Autopilot Interval"), min_value=10, max_value=720, value=current.interval_minutes, step=10
        )

        niche = st.text_area(
            tr("Autopilot Niche"), value=current.niche, max_chars=300, height=80,
            help=tr("Autopilot Niche Help"),
        )

        col_lang, col_voice = st.columns([1, 2])
        video_language = col_lang.text_input(
            tr("Autopilot Language"), value=current.video_language, max_chars=10
        )
        voice_name = col_voice.text_input(
            tr("Autopilot Voice"), value=current.voice_name, max_chars=100,
            help=tr("Autopilot Voice Help"),
        )

        col_rate, col_font = st.columns(2)
        voice_rate = col_rate.slider(
            tr("Autopilot Voice Rate"), min_value=0.5, max_value=2.0, value=float(current.voice_rate), step=0.05
        )
        font_size = col_font.number_input(
            tr("Autopilot Font Size"), min_value=30, max_value=120, value=current.font_size, step=1
        )

        col_attempts, col_delay, col_quota = st.columns(3)
        max_attempts = col_attempts.number_input(
            tr("Autopilot Max Attempts"), min_value=1, max_value=5, value=current.max_attempts, step=1
        )
        retry_delay = col_delay.number_input(
            tr("Autopilot Retry Delay"), min_value=1, max_value=120, value=current.retry_delay_minutes, step=1
        )
        quota = col_quota.number_input(
            tr("Autopilot Daily Quota"), min_value=1, max_value=50, value=current.daily_upload_quota, step=1,
            help=tr("Autopilot Daily Quota Help"),
        )

        fact_check = st.toggle(
            tr("Autopilot Fact Check"), value=current.fact_check_enabled, help=tr("Autopilot Fact Check Help")
        )
        metrics = st.toggle(
            tr("Autopilot Metrics"), value=current.metrics_enabled, help=tr("Autopilot Metrics Help")
        )

        submitted = st.form_submit_button(
            tr("Autopilot Save"), type="primary", icon=":material/save:", width="stretch"
        )

    if submitted:
        try:
            saved = autopilot.save_settings(
                {
                    "enabled": enabled,
                    "videos_per_day": int(videos_per_day),
                    "start_time": start_time.strftime("%H:%M"),
                    "interval_minutes": int(interval_minutes),
                    "niche": niche.strip(),
                    "video_language": video_language.strip(),
                    "voice_name": voice_name.strip(),
                    "voice_rate": float(voice_rate),
                    "font_size": int(font_size),
                    "max_attempts": int(max_attempts),
                    "retry_delay_minutes": int(retry_delay),
                    "daily_upload_quota": int(quota),
                    "fact_check_enabled": fact_check,
                    "metrics_enabled": metrics,
                }
            )
        except ValueError as exc:
            st.error(tr("Autopilot Save Failed").format(error=exc), icon=":material/error:")
        else:
            st.success(tr("Autopilot Saved"), icon=":material/check_circle:")
            current = saved

    slots = autopilot.slot_times(current, datetime.now().date())
    st.caption(
        tr("Autopilot Slots Preview").format(
            slots=", ".join(slot.strftime("%H:%M") for slot in slots) or "-"
        )
    )


# --- LLM fallback (inside the LLM settings tab) ---------------------------------


def render_llm_fallback(tr, container, set_config, app_config) -> None:
    with container.container(border=True):
        st.markdown(f"**{tr('Autopilot Fallback Title')}**")
        st.caption(tr("Autopilot Fallback Help"))
        fallback = st.toggle(
            tr("Autopilot Fallback Enable"),
            value=bool(app_config.get("openrouter_free_fallback", True)),
            key="openrouter_free_fallback_toggle",
        )
        set_config("app", "openrouter_free_fallback", fallback)
        free_only = st.toggle(
            tr("Autopilot Free Only"),
            value=bool(app_config.get("llm_free_only", True)),
            key="llm_free_only_toggle",
        )
        set_config("app", "llm_free_only", free_only)

        if st.button(
            tr("Autopilot Show Models"),
            key="show_free_models_button",
            icon=":material/list:",
            width="stretch",
        ):
            with st.spinner(tr("Autopilot Loading Models")):
                try:
                    rows = llm_free_models.status(
                        llm_free_models.preferred_model(app_config), free_only
                    )
                except Exception as exc:
                    st.error(tr("Autopilot Load Failed").format(error=exc))
                    return
            st.dataframe(
                [
                    {
                        "order": index + 1,
                        "model": row["id"],
                        "state": tr(_MODEL_STATE_KEYS.get(row["state"], row["state"])),
                        "context": row["context_length"],
                    }
                    for index, row in enumerate(rows)
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "order": st.column_config.NumberColumn("#", width="small"),
                    "model": st.column_config.TextColumn(tr("Autopilot Col Model"), width="large"),
                    "context": st.column_config.NumberColumn(tr("Autopilot Col Context"), format="%d", width="small"),
                    "state": st.column_config.TextColumn(tr("Autopilot Col Status"), width="small"),
                },
            )
