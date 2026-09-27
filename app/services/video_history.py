"""Permanent history of every generated video, whatever started it.

Lives in the same sqlite file as ``schedule_store`` (one local database,
survives restarts, shared by the API and WebUI processes). It is the single
source of truth for: repeat-avoidance (``fact_key``), the YouTube daily
upload quota, per-video metrics, and the autopilot's small key/value state.
"""

import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import date, datetime, timedelta, timezone

from app.services import schedule_store

STATUS_GENERATING = "generating"
STATUS_GENERATED = "generated"
STATUS_PUBLISHED = "published"
STATUS_FAILED = "failed"
STATUS_REJECTED = "rejected"
DEFAULT_DAILY_UPLOAD_QUOTA = 6
# The YouTube Data API quota resets at midnight Pacific Time.
_QUOTA_TIMEZONE = "America/Los_Angeles"

STATUSES = (
    STATUS_GENERATING,
    STATUS_GENERATED,
    STATUS_PUBLISHED,
    STATUS_FAILED,
    STATUS_REJECTED,
)

_COLUMNS = (
    "id",
    "task_id",
    "subject",
    "fact_key",
    "status",
    "source",
    "occurrence_id",
    "attempt",
    "error",
    "llm_model",
    "youtube_id",
    "youtube_url",
    "views",
    "likes",
    "comments",
    "avg_view_percentage",
    "metrics_updated_at",
    "published_at",
    "created_at",
    "updated_at",
    "public_at",
    "views_24h",
    "views_7d",
    "tts_voice",
)
# Added after the first release; ``_connect`` adds them to older databases.
_ADDED_COLUMNS = (
    ("public_at", "REAL"),
    ("views_24h", "INTEGER"),
    ("views_7d", "INTEGER"),
    ("tts_voice", "TEXT"),
)
_DAY = 86400.0
# Views at a fixed age, so a new video is compared fairly with an old one.
SNAPSHOT_WINDOWS = (("views_24h", _DAY), ("views_7d", 7 * _DAY))
# A snapshot taken later than this is too far from its window to be
# comparable, so it stays empty (e.g. videos from before this feature).
SNAPSHOT_GRACE_SECONDS = _DAY
# Upload time bound for snapshot candidates; covers a scheduled publishAt.
_SNAPSHOT_CANDIDATE_MAX_AGE = 14 * _DAY


def _connect(db_path: str | None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or schedule_store._default_db_path(), timeout=30)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL UNIQUE,
            subject TEXT NOT NULL,
            fact_key TEXT,
            status TEXT NOT NULL,
            source TEXT NOT NULL,
            occurrence_id INTEGER,
            attempt INTEGER NOT NULL DEFAULT 1,
            error TEXT,
            llm_model TEXT,
            youtube_id TEXT,
            youtube_url TEXT,
            views INTEGER,
            likes INTEGER,
            comments INTEGER,
            avg_view_percentage REAL,
            metrics_updated_at REAL,
            published_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_videos_fact_key ON videos (fact_key);
        CREATE INDEX IF NOT EXISTS idx_videos_youtube_id ON videos (youtube_id);
        CREATE INDEX IF NOT EXISTS idx_videos_status_created
            ON videos (status, created_at);
        CREATE TABLE IF NOT EXISTS youtube_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            uploaded_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_youtube_uploads_at
            ON youtube_uploads (uploaded_at);
        CREATE TABLE IF NOT EXISTS autopilot_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS video_comments (
            comment_id TEXT PRIMARY KEY,
            youtube_id TEXT NOT NULL,
            author_channel_id TEXT,
            text TEXT NOT NULL,
            like_count INTEGER NOT NULL DEFAULT 0,
            published_at REAL,
            synced_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_video_comments_published
            ON video_comments (published_at);
        """
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(videos)")}
    for name, kind in _ADDED_COLUMNS:
        if name in existing:
            continue
        try:
            conn.execute(f"ALTER TABLE videos ADD COLUMN {name} {kind}")
        except sqlite3.OperationalError as exc:
            # Another process added it between the PRAGMA and the ALTER.
            if "duplicate column" not in str(exc):
                raise
    return conn


def normalize_fact_key(fact_key: str | None) -> str:
    return (fact_key or "").strip().casefold()


def _row_to_dict(row: tuple) -> dict:
    return dict(zip(_COLUMNS, row))


def record(
    task_id: str,
    subject: str,
    source: str,
    *,
    fact_key: str | None = None,
    occurrence_id: int | None = None,
    attempt: int = 1,
    llm_model: str = "",
    status: str = STATUS_GENERATING,
    overwrite: bool = True,
    created_at: float | None = None,
    db_path: str | None = None,
) -> None:
    now = time.time()
    on_conflict = (
        """DO UPDATE SET
                subject = excluded.subject,
                fact_key = COALESCE(excluded.fact_key, videos.fact_key),
                status = excluded.status,
                updated_at = excluded.updated_at"""
        if overwrite
        else "DO NOTHING"
    )
    with closing(_connect(db_path)) as conn:
        conn.execute(
            f"""
            INSERT INTO videos (
                task_id, subject, fact_key, status, source, occurrence_id,
                attempt, llm_model, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) {on_conflict}
            """,
            (
                task_id,
                subject,
                normalize_fact_key(fact_key) or None,
                status,
                source,
                occurrence_id,
                attempt,
                llm_model or None,
                created_at or now,
                now,
            ),
        )
        conn.commit()


def set_status(
    task_id: str, status: str, error: str | None = None, db_path: str | None = None
) -> None:
    with closing(_connect(db_path)) as conn:
        conn.execute(
            "UPDATE videos SET status = ?, error = ?, updated_at = ? WHERE task_id = ?",
            (status, error, time.time(), task_id),
        )
        conn.commit()


def set_tts_voice(task_id: str, tts_voice: str, db_path: str | None = None) -> None:
    """Voice that narrated the task; no-op for tasks outside the history."""
    with closing(_connect(db_path)) as conn:
        conn.execute(
            "UPDATE videos SET tts_voice = ?, updated_at = ? WHERE task_id = ?",
            (tts_voice, time.time(), task_id),
        )
        conn.commit()


def mark_published(
    task_id: str,
    youtube_id: str,
    youtube_url: str,
    *,
    subject: str = "",
    db_path: str | None = None,
) -> None:
    """Flag a task as published; creates the row for tasks nobody recorded."""
    now = time.time()
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO videos (
                task_id, subject, status, source, youtube_id, youtube_url,
                published_at, created_at, updated_at
            ) VALUES (?, ?, ?, 'unknown', ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                status = excluded.status,
                error = NULL,
                youtube_id = excluded.youtube_id,
                youtube_url = excluded.youtube_url,
                published_at = excluded.published_at,
                updated_at = excluded.updated_at
            """,
            (
                task_id,
                subject or task_id,
                STATUS_PUBLISHED,
                youtube_id,
                youtube_url,
                now,
                now,
                now,
            ),
        )
        conn.commit()


def fact_key_exists(fact_key: str, db_path: str | None = None) -> bool:
    """A failed render never reached the channel, so its fact stays free."""
    key = normalize_fact_key(fact_key)
    if not key:
        return False
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT 1 FROM videos WHERE fact_key = ? AND status != ? LIMIT 1",
            (key, STATUS_FAILED),
        ).fetchone()
    return row is not None


def known_topics(limit: int = 300, db_path: str | None = None) -> list[tuple]:
    """``(subject, fact_key)`` of every non-failed video, newest first."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT subject, fact_key FROM videos WHERE status != ? "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (STATUS_FAILED, limit),
        ).fetchall()
    return [tuple(row) for row in rows]


def list_videos(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    db_path: str | None = None,
) -> tuple[list[dict], int]:
    where, args = ("WHERE status = ?", [status]) if status else ("", [])
    with closing(_connect(db_path)) as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM videos {where}", args).fetchone()[0]
        rows = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM videos {where} "
            "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            [*args, limit, offset],
        ).fetchall()
    return [_row_to_dict(row) for row in rows], total


def record_upload(
    video_id: str, uploaded_at: float | None = None, db_path: str | None = None
) -> None:
    with closing(_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO youtube_uploads (video_id, uploaded_at) VALUES (?, ?)",
            (video_id, uploaded_at if uploaded_at is not None else time.time()),
        )
        conn.commit()


def uploads_since(timestamp: float, db_path: str | None = None) -> int:
    with closing(_connect(db_path)) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM youtube_uploads WHERE uploaded_at >= ?",
            (timestamp,),
        ).fetchone()[0]


def published_youtube_ids(limit: int = 500, db_path: str | None = None) -> list[str]:
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT youtube_id FROM videos WHERE status = ? AND youtube_id IS NOT NULL "
            "ORDER BY published_at DESC LIMIT ?",
            (STATUS_PUBLISHED, limit),
        ).fetchall()
    return [row[0] for row in rows]


def update_metrics(
    youtube_id: str,
    *,
    views: int | None = None,
    likes: int | None = None,
    comments: int | None = None,
    avg_view_percentage: float | None = None,
    public_at: float | None = None,
    now: float | None = None,
    db_path: str | None = None,
) -> None:
    """``public_at`` is when YouTube made the video public; with it, the
    current ``views`` also fill each snapshot window the video is inside."""
    now = time.time() if now is None else now
    snapshot = {column: None for column, _ in SNAPSHOT_WINDOWS}
    if public_at is not None and views is not None:
        age = now - public_at
        for column, window in SNAPSHOT_WINDOWS:
            if window <= age < window + SNAPSHOT_GRACE_SECONDS:
                snapshot[column] = views
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """
            UPDATE videos SET
                views = COALESCE(?, views),
                likes = COALESCE(?, likes),
                comments = COALESCE(?, comments),
                avg_view_percentage = COALESCE(?, avg_view_percentage),
                public_at = COALESCE(?, public_at),
                views_24h = COALESCE(views_24h, ?),
                views_7d = COALESCE(views_7d, ?),
                metrics_updated_at = ?
            WHERE youtube_id = ?
            """,
            (
                views,
                likes,
                comments,
                avg_view_percentage,
                public_at,
                snapshot["views_24h"],
                snapshot["views_7d"],
                now,
                youtube_id,
            ),
        )
        conn.commit()


def snapshot_candidates(now: float | None = None, db_path: str | None = None) -> list[str]:
    """Published videos that may still need a snapshot window filled."""
    now = time.time() if now is None else now
    last_window = SNAPSHOT_WINDOWS[-1][1] + SNAPSHOT_GRACE_SECONDS
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT youtube_id FROM videos WHERE status = ? AND youtube_id IS NOT NULL "
            "AND published_at >= ? AND (views_24h IS NULL OR views_7d IS NULL) "
            "AND (public_at IS NULL OR public_at >= ?) ORDER BY published_at DESC",
            (STATUS_PUBLISHED, now - _SNAPSHOT_CANDIDATE_MAX_AGE, now - last_window),
        ).fetchall()
    return [row[0] for row in rows]


def performers(limit: int = 5, db_path: str | None = None) -> tuple[list, list]:
    """Best and worst published videos by views in their first 24h, for the
    topic prompt. Each side gets at most half the measured videos, so the
    same video is never both a best and a worst example."""
    where = "WHERE status = ? AND views_24h IS NOT NULL"
    query = (
        f"SELECT {', '.join(_COLUMNS)} FROM videos {where} "
        "ORDER BY views_24h {order}, id {order} LIMIT ?"
    )
    with closing(_connect(db_path)) as conn:
        measured = conn.execute(
            f"SELECT COUNT(*) FROM videos {where}", (STATUS_PUBLISHED,)
        ).fetchone()[0]
        size = min(limit, measured // 2)
        if size == 0:
            return [], []
        top = conn.execute(query.format(order="DESC"), (STATUS_PUBLISHED, size)).fetchall()
        bottom = conn.execute(query.format(order="ASC"), (STATUS_PUBLISHED, size)).fetchall()
    return [_row_to_dict(r) for r in top], [_row_to_dict(r) for r in bottom]



def public_youtube_ids_since(since: float, db_path: str | None = None) -> list[str]:
    """Published videos YouTube made public at or after ``since``."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT youtube_id FROM videos WHERE status = ? AND youtube_id IS NOT NULL "
            "AND public_at >= ? ORDER BY public_at DESC",
            (STATUS_PUBLISHED, since),
        ).fetchall()
    return [row[0] for row in rows]


_COMMENT_COLUMNS = (
    "comment_id",
    "youtube_id",
    "author_channel_id",
    "text",
    "like_count",
    "published_at",
)


def add_comments(comments: list[dict], db_path: str | None = None) -> int:
    """Insert comments not seen yet; returns how many were new."""
    if not comments:
        return 0
    now = time.time()
    with closing(_connect(db_path)) as conn:
        before = conn.total_changes
        conn.executemany(
            f"INSERT OR IGNORE INTO video_comments ({', '.join(_COMMENT_COLUMNS)}, synced_at) "
            f"VALUES ({', '.join('?' * (len(_COMMENT_COLUMNS) + 1))})",
            [tuple(c.get(col) for col in _COMMENT_COLUMNS) + (now,) for c in comments],
        )
        conn.commit()
        return conn.total_changes - before


def comments_since(since: float, limit: int = 200, db_path: str | None = None) -> list[dict]:
    """Comments published at or after ``since``, newest first."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            f"SELECT {', '.join(_COMMENT_COLUMNS)} FROM video_comments "
            "WHERE published_at >= ? ORDER BY published_at DESC, comment_id LIMIT ?",
            (since, limit),
        ).fetchall()
    return [dict(zip(_COMMENT_COLUMNS, row)) for row in rows]


def get_state(key: str, default=None, db_path: str | None = None):
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT value FROM autopilot_state WHERE key = ?", (key,)
        ).fetchone()
    return default if row is None else json.loads(row[0])


def set_state(key: str, value, db_path: str | None = None) -> None:
    with closing(_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO autopilot_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        conn.commit()


def quota_day_start(now: datetime | None = None) -> float:
    """Epoch seconds of the current YouTube quota day's start (midnight PT)."""
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(_QUOTA_TIMEZONE)
    except Exception:
        # ponytail: no tzdata in the image -> fixed PST, off by 1h during DST.
        tz = timezone(timedelta(hours=-8))
    local = (now or datetime.now(timezone.utc)).astimezone(tz)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def daily_upload_quota() -> int:
    settings = get_state("settings") or {}
    return int(settings.get("daily_upload_quota") or DEFAULT_DAILY_UPLOAD_QUOTA)


def backup_database(
    dest_dir: str,
    keep: int = 14,
    today: date | None = None,
    db_path: str | None = None,
) -> str:
    """Online copy of the whole sqlite file (schedule + history + state),
    keeping only the newest ``keep`` daily files."""
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, f"schedule-{(today or date.today()).isoformat()}.db")
    with closing(_connect(db_path)) as source, closing(sqlite3.connect(target)) as dest:
        source.backup(dest)
    backups = sorted(
        name
        for name in os.listdir(dest_dir)
        if name.startswith("schedule-") and name.endswith(".db")
    )
    for old in backups[:-keep]:
        os.remove(os.path.join(dest_dir, old))
    return target
