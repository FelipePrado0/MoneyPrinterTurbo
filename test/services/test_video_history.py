import time
from contextlib import closing

import pytest

from app.services import video_history as vh


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "schedule.db")


def test_record_then_list_newest_first(db):
    vh.record("t1", "Polvo tem 3 corações", "autopilot", fact_key="polvo:3-coracoes", db_path=db)
    vh.record("t2", "Relâmpago de 829 km", "webui", db_path=db)
    rows, total = vh.list_videos(db_path=db)
    assert total == 2
    assert [r["task_id"] for r in rows] == ["t2", "t1"]
    assert rows[1]["status"] == vh.STATUS_GENERATING
    assert rows[1]["fact_key"] == "polvo:3-coracoes"


def test_record_is_idempotent_per_task(db):
    vh.record("t1", "A", "api", db_path=db)
    vh.record("t1", "A", "api", db_path=db)
    assert vh.list_videos(db_path=db)[1] == 1


def test_fact_key_exists_ignores_failed_videos(db):
    vh.record("t1", "A", "autopilot", fact_key="polvo:3-coracoes", db_path=db)
    assert vh.fact_key_exists("polvo:3-coracoes", db_path=db)
    vh.set_status("t1", vh.STATUS_FAILED, "render broke", db_path=db)
    assert not vh.fact_key_exists("polvo:3-coracoes", db_path=db)
    vh.record("t2", "B", "autopilot", fact_key="x:y", db_path=db)
    vh.set_status("t2", vh.STATUS_REJECTED, "fact check", db_path=db)
    assert vh.fact_key_exists("x:y", db_path=db)


def test_fact_key_is_compared_normalized(db):
    vh.record("t1", "A", "autopilot", fact_key="Polvo:3-Coracoes ", db_path=db)
    assert vh.fact_key_exists("polvo:3-coracoes", db_path=db)


def test_mark_published_sets_youtube_fields(db):
    vh.record("t1", "A", "autopilot", db_path=db)
    vh.mark_published("t1", "yt123", "https://youtu.be/yt123", db_path=db)
    row = vh.list_videos(db_path=db)[0][0]
    assert row["status"] == vh.STATUS_PUBLISHED
    assert row["youtube_id"] == "yt123"
    assert row["published_at"] is not None


def test_mark_published_unknown_task_creates_row(db):
    vh.mark_published("ghost", "yt9", "u", subject="Manual", db_path=db)
    row = vh.list_videos(db_path=db)[0][0]
    assert row["task_id"] == "ghost" and row["subject"] == "Manual"


def test_uploads_since_counts_only_recent(db):
    vh.record_upload("a", uploaded_at=100.0, db_path=db)
    vh.record_upload("b", uploaded_at=200.0, db_path=db)
    assert vh.uploads_since(150.0, db_path=db) == 1


DAY = 86400.0


def _published(db, index, public_at=None, views_24h=None, now=None):
    vh.record(f"t{index}", f"S{index}", "autopilot", db_path=db)
    vh.mark_published(f"t{index}", f"y{index}", "u", db_path=db)
    if public_at is not None:
        vh.update_metrics(
            f"y{index}", views=views_24h, public_at=public_at, now=now, db_path=db
        )


def test_performers_rank_by_24h_views_without_overlap(db):
    now = 10 * DAY
    for i, views in enumerate([10, 500, 90]):
        _published(db, i, public_at=now - DAY - 60, views_24h=views, now=now)
    # Cumulative views must not decide the ranking any more.
    vh.update_metrics("y0", views=99999, db_path=db)
    assert set(vh.published_youtube_ids(db_path=db)) == {"y0", "y1", "y2"}
    top, bottom = vh.performers(limit=5, db_path=db)
    assert [r["subject"] for r in top] == ["S1"]
    assert [r["subject"] for r in bottom] == ["S0"]


def test_performers_empty_until_two_videos_have_24h_views(db):
    _published(db, 0, public_at=0.0, views_24h=50, now=DAY + 60)
    _published(db, 1)
    assert vh.performers(db_path=db) == ([], [])


def test_snapshots_are_taken_once_inside_each_window(db):
    public_at = 100 * DAY
    _published(db, 0)
    vh.update_metrics("y0", views=5, public_at=public_at, now=public_at + 3600, db_path=db)
    row = vh.list_videos(db_path=db)[0][0]
    assert row["public_at"] == public_at and row["views_24h"] is None

    vh.update_metrics("y0", views=100, public_at=public_at, now=public_at + DAY + 60, db_path=db)
    vh.update_metrics("y0", views=150, public_at=public_at, now=public_at + DAY + 3600, db_path=db)
    vh.update_metrics("y0", views=900, public_at=public_at, now=public_at + 7 * DAY + 60, db_path=db)
    row = vh.list_videos(db_path=db)[0][0]
    assert (row["views_24h"], row["views_7d"], row["views"]) == (100, 900, 900)


def test_snapshot_is_skipped_after_the_grace_period(db):
    _published(db, 0)
    vh.update_metrics("y0", views=700, public_at=0.0, now=3 * DAY, db_path=db)
    row = vh.list_videos(db_path=db)[0][0]
    assert row["views_24h"] is None and row["views_7d"] is None


def test_snapshot_needs_a_real_view_count(db):
    _published(db, 0)
    vh.update_metrics("y0", likes=3, public_at=0.0, now=DAY + 60, db_path=db)
    assert vh.list_videos(db_path=db)[0][0]["views_24h"] is None


def test_snapshot_candidates_only_recent_videos_missing_a_window(db):
    now = time.time()
    _published(db, 0)  # not public yet: public_at unknown
    _published(db, 1, public_at=now - 2 * DAY, views_24h=None, now=now)
    _published(db, 2, public_at=now - 30 * DAY, views_24h=None, now=now)
    _published(db, 3)  # both windows already filled
    vh.update_metrics("y3", views=1, public_at=now - DAY - 60, now=now, db_path=db)
    vh.update_metrics("y3", views=2, public_at=now - DAY - 60, now=now + 6 * DAY, db_path=db)
    _published(db, 4, public_at=now - DAY - 60, views_24h=40, now=now)  # 7d still due
    assert set(vh.snapshot_candidates(now=now, db_path=db)) == {"y0", "y1", "y4"}


def test_old_database_gets_the_new_columns(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.db")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "CREATE TABLE videos (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "task_id TEXT NOT NULL UNIQUE, subject TEXT NOT NULL, fact_key TEXT, "
            "status TEXT NOT NULL, source TEXT NOT NULL, occurrence_id INTEGER, "
            "attempt INTEGER NOT NULL DEFAULT 1, error TEXT, llm_model TEXT, "
            "youtube_id TEXT, youtube_url TEXT, views INTEGER, likes INTEGER, "
            "comments INTEGER, avg_view_percentage REAL, metrics_updated_at REAL, "
            "published_at REAL, created_at REAL NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO videos (task_id, subject, status, source, created_at, updated_at) "
            "VALUES ('old', 'Old', 'published', 'api', 1, 1)"
        )
        conn.commit()
    row = vh.list_videos(db_path=path)[0][0]
    assert row["subject"] == "Old" and row["views_24h"] is None


def test_known_topics_excludes_failed(db):
    vh.record("t1", "Keep", "autopilot", fact_key="k:1", db_path=db)
    vh.record("t2", "Drop", "autopilot", fact_key="k:2", db_path=db)
    vh.set_status("t2", vh.STATUS_FAILED, "x", db_path=db)
    assert vh.known_topics(db_path=db) == [("Keep", "k:1")]


def test_state_roundtrip(db):
    assert vh.get_state("paused", False, db_path=db) is False
    vh.set_state("paused", True, db_path=db)
    assert vh.get_state("paused", db_path=db) is True
    vh.set_state("settings", {"a": 1}, db_path=db)
    assert vh.get_state("settings", db_path=db) == {"a": 1}


def test_list_filters_by_status_and_paginates(db):
    for i in range(5):
        vh.record(f"t{i}", f"S{i}", "api", db_path=db)
        time.sleep(0.001)
    vh.set_status("t0", vh.STATUS_FAILED, "x", db_path=db)
    rows, total = vh.list_videos(limit=2, offset=1, db_path=db)
    assert total == 5 and len(rows) == 2
    failed, failed_total = vh.list_videos(status=vh.STATUS_FAILED, db_path=db)
    assert failed_total == 1 and failed[0]["task_id"] == "t0"


def test_publish_video_blocks_when_daily_quota_is_used(monkeypatch):
    from app.services import youtube_upload

    calls = []

    def fake_upload(**kwargs):
        calls.append(kwargs)
        return {"success": True, "video_id": f"id{len(calls)}", "url": "u"}

    monkeypatch.setattr(youtube_upload.youtube_upload_service, "upload_video", fake_upload)
    vh.set_state("settings", {"daily_upload_quota": 2})
    assert youtube_upload.publish_video("a.mp4", "t")["success"]
    assert youtube_upload.publish_video("b.mp4", "t")["success"]
    blocked = youtube_upload.publish_video("c.mp4", "t")
    assert not blocked["success"] and "quota" in blocked["error"]
    assert len(calls) == 2
    assert vh.uploads_since(vh.quota_day_start()) == 2


def test_quota_day_starts_at_pacific_midnight():
    from datetime import datetime, timezone

    start = vh.quota_day_start(datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc))
    assert datetime.fromtimestamp(start, timezone.utc) == datetime(
        2026, 9, 25, 7, 0, tzinfo=timezone.utc
    )


def test_backup_database_copies_and_rotates(db, tmp_path):
    import sqlite3
    from datetime import date

    vh.record("t1", "Kept", "api", db_path=db)
    backups = tmp_path / "backups"
    for day in range(1, 5):
        vh.backup_database(str(backups), keep=3, today=date(2026, 9, day), db_path=db)
    files = sorted(p.name for p in backups.iterdir())
    assert files == [
        "schedule-2026-09-02.db",
        "schedule-2026-09-03.db",
        "schedule-2026-09-04.db",
    ]
    with sqlite3.connect(backups / files[-1]) as conn:
        assert conn.execute("SELECT subject FROM videos").fetchone() == ("Kept",)


def test_record_without_overwrite_keeps_existing_row(db):
    vh.record("t1", "A", "autopilot", db_path=db)
    vh.mark_published("t1", "y", "u", db_path=db)
    vh.record("t1", "B", "schedule", status=vh.STATUS_GENERATED, overwrite=False, db_path=db)
    row = vh.list_videos(db_path=db)[0][0]
    assert row["status"] == vh.STATUS_PUBLISHED and row["subject"] == "A"
