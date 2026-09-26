"""Channel autopilot: plan, pick a new fact, verify, render, publish, learn.

Runs inside the API process on the existing schedule poller: every tick
``tick()`` makes sure today's slots exist as ordinary schedule occurrences
(group ``autopilot-YYYY-MM-DD``), and the poller hands each due one to
``dispatch()`` instead of the generic dispatcher. Nothing here publishes
directly: rendering and the YouTube upload stay in ``task.start()``, which
only gets an extra pre-publish quality check.
"""

import json
import os
import re
import subprocess
import unicodedata
from datetime import date, datetime, timedelta

from loguru import logger
from pydantic import ValidationError

from app.models import const
from app.models.schema import AutopilotSettings, VideoParams
from app.services import (
    llm,
    schedule_store,
    video_history,
    webhook_notifier,
    youtube_metrics,
)
from app.services import task as task_service
from app.services import video as video_service
from app.services.trend_topic import AUTO_TOPIC_SENTINEL, is_similar_to_recent
from app.utils import utils

GROUP_PREFIX = "autopilot-"
ATTEMPT_KEY = "autopilot_attempt"
TOPIC_ATTEMPTS = 3
KNOWN_TOPICS_IN_PROMPT = 300
METRICS_HOUR = 5
BACKUP_KEEP_DAYS = 14
MIN_VIDEO_SECONDS = 5
# YouTube Shorts accepts up to 3 minutes.
MAX_VIDEO_SECONDS = 180
DECODE_TIMEOUT_SECONDS = 300

# Fixed rendering template validated on the channel (see VIDEO_PLAYBOOK.md);
# only the fields exposed in AutopilotSettings vary.
_PARAMS_TEMPLATE = {
    "video_aspect": "9:16",
    "video_fit_mode": "cover",
    "video_concat_mode": "random",
    "video_clip_duration": 3,
    "video_count": 1,
    "video_source": "pexels",
    "voice_volume": 1.0,
    "bgm_type": "custom",
    "bgm_file": "",
    "bgm_volume": 0.2,
    "subtitle_enabled": True,
    "subtitle_position": "bottom",
    "font_name": "MicrosoftYaHeiBold.ttc",
    "text_fore_color": "#FFFFFF",
    "stroke_color": "#000000",
    "stroke_width": 0.0,
    "youtube_review_required": False,
}


class AutopilotError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now()


# --- settings & state -------------------------------------------------------


def load_settings() -> AutopilotSettings:
    stored = video_history.get_state("settings") or {}
    try:
        return AutopilotSettings(**stored)
    except ValidationError as exc:
        logger.error(f"stored autopilot settings are invalid, using defaults: {exc}")
        return AutopilotSettings()


def save_settings(update: dict, now: datetime | None = None) -> AutopilotSettings:
    """Validate and persist a partial update, then rebuild today's plan."""
    merged = {**load_settings().model_dump()}
    merged.update({k: v for k, v in update.items() if v is not None})
    try:
        settings = AutopilotSettings(**merged)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    video_history.set_state("settings", settings.model_dump())
    replan_today(now=now)
    return settings


def is_paused() -> bool:
    return bool(video_history.get_state("paused", False))


def pause(now: datetime | None = None) -> None:
    video_history.set_state("paused", True)
    now = now or _now()
    schedule_store.cancel_group(_group_id(now.date()))


def resume(now: datetime | None = None) -> None:
    video_history.set_state("paused", False)
    replan_today(now=now)


# --- planning -----------------------------------------------------------------


def _group_id(day: date) -> str:
    return f"{GROUP_PREFIX}{day.isoformat()}"


def is_autopilot_occurrence(occurrence: dict) -> bool:
    return str(occurrence.get("group_id", "")).startswith(GROUP_PREFIX)


def slot_times(settings: AutopilotSettings, day: date) -> list[datetime]:
    hour, minute = (int(part) for part in settings.start_time.split(":"))
    first = datetime(day.year, day.month, day.day, hour, minute)
    slots = [
        first + timedelta(minutes=settings.interval_minutes * index)
        for index in range(settings.videos_per_day)
    ]
    return [slot for slot in slots if slot.date() == day]


def build_params(settings: AutopilotSettings) -> dict:
    return {
        **_PARAMS_TEMPLATE,
        "video_subject": AUTO_TOPIC_SENTINEL,
        "video_language": settings.video_language,
        "voice_name": settings.voice_name,
        "voice_rate": settings.voice_rate,
        "font_size": settings.font_size,
    }


def plan_day(now: datetime | None = None, settings: AutopilotSettings | None = None) -> int:
    """Create today's remaining slots once per day. Returns how many."""
    now = now or _now()
    settings = settings or load_settings()
    if not settings.enabled or is_paused():
        return 0
    today = now.date().isoformat()
    if video_history.get_state("planned_date") == today:
        return 0
    video_history.set_state("planned_date", today)
    slots = [slot for slot in slot_times(settings, now.date()) if slot > now]
    if not slots:
        return 0
    schedule_store.create_schedule(
        [{"generate_at": slot, "video_subject": AUTO_TOPIC_SENTINEL} for slot in slots],
        build_params(settings),
        group_id=_group_id(now.date()),
    )
    logger.info(f"autopilot planned {len(slots)} videos for {today}")
    return len(slots)


def replan_today(now: datetime | None = None) -> int:
    now = now or _now()
    schedule_store.cancel_group(_group_id(now.date()))
    video_history.set_state("planned_date", None)
    return plan_day(now=now)


# --- topic & fact check -------------------------------------------------------


def slugify_fact_key(value: str) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode()
    )
    parts = [re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-") for part in ascii_text.split(":")]
    return ":".join(part for part in parts if part)


def _parse_json_object(text: str) -> dict | None:
    if not text or text.startswith("Error: "):
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _topic_prompt(settings: AutopilotSettings, rejected: list[str]) -> str:
    known = video_history.known_topics(limit=KNOWN_TOPICS_IN_PROMPT)
    known_lines = "\n".join(
        f"- {subject}" + (f" [{key}]" if key else "") for subject, key in known
    ) or "- (none yet)"
    top, bottom = video_history.performers(limit=5)
    performance = ""
    if top:
        performance = "\n## What performed best (make more like these)\n" + "\n".join(
            f"- {row['subject']} ({row['views_24h']} views in the first 24h)" for row in top
        )
        performance += "\n## What performed worst (avoid this angle)\n" + "\n".join(
            f"- {row['subject']} ({row['views_24h']} views in the first 24h)" for row in bottom
        )
    rejected_block = ""
    if rejected:
        rejected_block = "\n## Already rejected in this round, never reuse\n" + "\n".join(
            f"- {item}" for item in rejected
        )
    return f"""
# Role: YouTube Shorts researcher and scriptwriter

## Niche
{settings.niche}

## Task
Pick ONE surprising, well-established and verifiable fact for a new Short and
write it. Rules:
1. The fact must be scientifically correct and verifiable in reliable sources
   (NASA, universities, science outlets). Never invent numbers or events.
2. It must be shown with generic stock footage (space, ocean, animals, nature,
   weather, human body, cities). Avoid specific people, brands or one-off news.
3. It must NOT repeat any fact or angle from the history below.

## Output: only this JSON object, nothing else
{{"subject": "catchy title in {settings.video_language}, strong hook, max 90 characters",
 "fact_key": "entity:fact in lowercase english slug, e.g. octopus:three-hearts",
 "script": "narration in {settings.video_language}, 120-200 words: hook, fact explained, final call to comment",
 "terms": "4-6 generic English stock-footage search terms, comma separated"}}

## History already published (never repeat)
{known_lines}
{performance}{rejected_block}
""".strip()


def generate_topic(settings: AutopilotSettings) -> dict:
    """Ask the LLM for a new fact, rejecting duplicates by ``fact_key``/title."""
    rejected: list[str] = []
    known_subjects = [subject for subject, _ in video_history.known_topics()]
    for _ in range(TOPIC_ATTEMPTS):
        reply = llm._generate_response(_topic_prompt(settings, rejected))
        data = _parse_json_object(reply)
        if not data:
            rejected.append(f"(invalid answer: {str(reply)[:80]})")
            continue
        subject = str(data.get("subject", "")).strip()[:100]
        fact_key = slugify_fact_key(str(data.get("fact_key", "")))
        script = str(data.get("script", "")).strip()
        terms = data.get("terms", "")
        if isinstance(terms, list):
            terms = ", ".join(str(term) for term in terms)
        terms = str(terms).strip()
        if not (subject and fact_key and script and terms):
            rejected.append(f"(incomplete answer: {subject or fact_key})")
            continue
        if video_history.fact_key_exists(fact_key) or is_similar_to_recent(
            subject, known_subjects
        ):
            rejected.append(f"{subject} [{fact_key}]")
            continue
        return {
            "subject": subject,
            "fact_key": fact_key,
            "script": script,
            "terms": terms,
            "llm_model": llm.get_last_used_model(),
        }
    raise AutopilotError(f"no new topic after {TOPIC_ATTEMPTS} attempts: {rejected}")


def fact_check(subject: str, script: str) -> tuple[bool, str]:
    """Strict LLM review; anything but an explicit approval is a rejection."""
    prompt = f"""
# Role: strict science fact-checker

Review the title and narration below. Approve ONLY if every factual claim is
well-established and correctly stated. Reject invented or wrong numbers,
dates or events, exaggerations that change the meaning, and unverifiable
claims.

## Title
{subject}

## Narration
{script}

## Output: only this JSON object
{{"approved": true or false, "reason": "short explanation when rejected"}}
""".strip()
    data = _parse_json_object(llm._generate_response(prompt))
    if not data:
        return False, "fact check returned no valid verdict"
    approved = data.get("approved") is True
    return approved, "" if approved else str(data.get("reason") or "rejected")


# --- rendered video check -----------------------------------------------------


def _decodes_cleanly(path: str) -> bool:
    try:
        result = subprocess.run(
            [utils.get_ffmpeg_binary(), "-nostdin", "-v", "error", "-xerror",
             "-i", path, "-f", "null", "-"],
            capture_output=True,
            timeout=DECODE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"ffmpeg decode check could not run: {exc}")
        return False
    return result.returncode == 0


def check_rendered_video(
    video_paths: list[str], subtitle_path: str, subtitle_enabled: bool
) -> str | None:
    """Return why the render must not be published, or ``None`` when fine."""
    if subtitle_enabled:
        try:
            with open(subtitle_path, encoding="utf-8") as subtitle_file:
                if not subtitle_file.read().strip():
                    return "subtitle file is empty"
        except (OSError, TypeError, ValueError):
            return "subtitle file is missing"
    for path in video_paths:
        try:
            with video_service._open_video_clip_quietly(path, audio=True) as clip:
                duration, has_audio = float(clip.duration), clip.audio is not None
        except Exception as exc:
            return f"video cannot be opened: {path}: {exc}"
        if duration < MIN_VIDEO_SECONDS:
            return f"video too short ({duration:.1f}s): {path}"
        if duration > MAX_VIDEO_SECONDS:
            return f"video too long for Shorts ({duration:.1f}s): {path}"
        if not has_audio:
            return f"video has no audio track: {path}"
        if not _decodes_cleanly(path):
            return f"video does not decode cleanly: {path}"
    return None


# --- dispatch -----------------------------------------------------------------


def uploads_today() -> int:
    return video_history.uploads_since(video_history.quota_day_start())


def _retry_or_alert(occurrence: dict, attempt: int, settings, reason: str) -> None:
    schedule_store.mark_failed(occurrence["id"], reason)
    if attempt < settings.max_attempts:
        params = {**occurrence["params"], "video_subject": AUTO_TOPIC_SENTINEL}
        params[ATTEMPT_KEY] = attempt + 1
        schedule_store.create_schedule(
            [
                {
                    "generate_at": _now() + timedelta(minutes=settings.retry_delay_minutes),
                    "video_subject": AUTO_TOPIC_SENTINEL,
                }
            ],
            params,
            group_id=occurrence["group_id"],
        )
        logger.warning(
            f"autopilot attempt {attempt}/{settings.max_attempts} failed, "
            f"retrying in {settings.retry_delay_minutes} min: {reason}"
        )
        return
    logger.error(f"autopilot slot failed after {attempt} attempts: {reason}")
    webhook_notifier.notify_event(
        "autopilot.slot_failed",
        {
            "occurrence_id": occurrence["id"],
            "attempts": attempt,
            "error": reason,
        },
    )


def dispatch(occurrence: dict) -> None:
    """Run one claimed autopilot slot end to end. Never raises."""
    try:
        _dispatch(occurrence)
    except Exception as exc:
        logger.exception(f"autopilot dispatch crashed: {exc}")
        schedule_store.mark_failed(occurrence["id"], f"autopilot crashed: {exc}")


def _dispatch(occurrence: dict) -> None:
    occurrence_id, task_id = occurrence["id"], occurrence["task_id"]
    attempt = int(occurrence["params"].get(ATTEMPT_KEY) or 1)
    settings = load_settings()
    if not settings.enabled or is_paused():
        schedule_store.mark_failed(occurrence_id, "autopilot disabled or paused")
        return
    if uploads_today() >= settings.daily_upload_quota:
        schedule_store.mark_failed(occurrence_id, "daily YouTube upload quota reached")
        return

    try:
        topic = generate_topic(settings)
    except AutopilotError as exc:
        _retry_or_alert(occurrence, attempt, settings, f"topic: {exc}")
        return

    subject = topic["subject"]
    schedule_store.update_video_subject(occurrence_id, subject)
    video_history.record(
        task_id,
        subject,
        "autopilot",
        fact_key=topic["fact_key"],
        occurrence_id=occurrence_id,
        attempt=attempt,
        llm_model=topic["llm_model"],
    )

    if settings.fact_check_enabled:
        approved, reason = fact_check(subject, topic["script"])
        if not approved:
            video_history.set_status(task_id, video_history.STATUS_REJECTED, reason)
            _retry_or_alert(occurrence, attempt, settings, f"fact check: {reason}")
            return

    raw_params = {k: v for k, v in occurrence["params"].items() if k != ATTEMPT_KEY}
    params = VideoParams(
        **{
            **raw_params,
            "video_subject": subject,
            "video_script": topic["script"],
            "video_terms": topic["terms"],
        }
    )
    result = task_service.start(
        task_id,
        params,
        pre_publish_check=lambda paths, subtitle: check_rendered_video(
            paths, subtitle, bool(params.subtitle_enabled)
        ),
    )
    if isinstance(result, dict) and result.get("state") == const.TASK_STATE_FAILED:
        stage, error = result.get("failed_stage"), str(result.get("error") or "")
        status = (
            video_history.STATUS_REJECTED
            if stage == task_service.QUALITY_GATE_STAGE
            else video_history.STATUS_FAILED
        )
        video_history.set_status(task_id, status, error)
        _retry_or_alert(occurrence, attempt, settings, f"{stage}: {error}")


# --- periodic work ------------------------------------------------------------


def run_metrics() -> None:
    video_ids = video_history.published_youtube_ids()
    metrics, problems = youtube_metrics.fetch_metrics(video_ids)
    for video_id, values in metrics.items():
        video_history.update_metrics(video_id, **values)
    video_history.set_state("metrics_problems", problems)
    video_history.set_state("metrics_updated_at", _now().isoformat(timespec="seconds"))
    logger.info(f"youtube metrics updated: {len(metrics)} videos, problems: {problems}")


def run_snapshots() -> None:
    """Hourly: fill the 24h/7d view snapshots of recently published videos."""
    video_ids = video_history.snapshot_candidates()
    if not video_ids:
        return
    metrics, problems = youtube_metrics.fetch_metrics(video_ids, statistics_only=True)
    for video_id, values in metrics.items():
        video_history.update_metrics(video_id, **values)
    if problems:
        # Only report here; the daily run_metrics clears them once it works.
        video_history.set_state("metrics_problems", problems)


def run_backup() -> None:
    target = video_history.backup_database(
        os.path.join(utils.storage_dir(create=True), "backups"), keep=BACKUP_KEEP_DAYS
    )
    logger.info(f"schedule database backed up: {target}")


def backfill_history_from_schedule() -> int:
    """Seed the history with topics dispatched before it existed, so the
    repeat check also knows what was published by the old scheduler."""
    added = 0
    for occurrence in schedule_store.list_occurrences(status=schedule_store.STATUS_DISPATCHED):
        subject = occurrence["video_subject"]
        if not occurrence["task_id"] or subject == AUTO_TOPIC_SENTINEL:
            continue
        video_history.record(
            occurrence["task_id"],
            subject,
            "schedule",
            status=video_history.STATUS_GENERATED,
            overwrite=False,
            created_at=occurrence["generate_at"].timestamp(),
        )
        added += 1
    return added


def tick(now: datetime, submit) -> None:
    """Called by the schedule poller every few seconds."""
    if not video_history.get_state("history_backfilled"):
        video_history.set_state("history_backfilled", True)
        logger.info(f"video history backfilled: {backfill_history_from_schedule()} topics")
    settings = load_settings()
    plan_day(now=now, settings=settings)
    today = now.date().isoformat()
    if (
        settings.metrics_enabled
        and now.hour >= METRICS_HOUR
        and video_history.get_state("metrics_date") != today
    ):
        video_history.set_state("metrics_date", today)
        submit(run_metrics)
    hour = now.strftime("%Y-%m-%dT%H")
    if settings.metrics_enabled and video_history.get_state("snapshot_hour") != hour:
        video_history.set_state("snapshot_hour", hour)
        submit(run_snapshots)
    if video_history.get_state("backup_date") != today:
        video_history.set_state("backup_date", today)
        submit(run_backup)


def status(now: datetime | None = None) -> dict:
    now = now or _now()
    settings = load_settings()
    today = schedule_store.list_occurrences(group_id=_group_id(now.date()))
    upcoming = [
        o for o in today if o["status"] == schedule_store.STATUS_PENDING and o["generate_at"] > now
    ]
    return {
        "settings": settings.model_dump(),
        "paused": is_paused(),
        "uploads_today": uploads_today(),
        "next_run": upcoming[0]["generate_at"].isoformat() if upcoming else None,
        "today": [
            {
                "id": o["id"],
                "generate_at": o["generate_at"].isoformat(),
                "subject": o["video_subject"],
                "status": o["status"],
                "attempt": int(o["params"].get(ATTEMPT_KEY) or 1),
                "error": o["error"],
                "task_id": o["task_id"],
            }
            for o in today
            if o["status"] != schedule_store.STATUS_CANCELLED
        ],
        "metrics_problems": video_history.get_state("metrics_problems") or [],
        "metrics_updated_at": video_history.get_state("metrics_updated_at"),
    }
