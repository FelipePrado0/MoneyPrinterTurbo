import time

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


def test_metrics_and_performers(db):
    for i, views in enumerate([10, 500, 90]):
        vh.record(f"t{i}", f"S{i}", "autopilot", db_path=db)
        vh.mark_published(f"t{i}", f"y{i}", "u", db_path=db)
        vh.update_metrics(f"y{i}", views=views, likes=1, comments=0, db_path=db)
    assert set(vh.published_youtube_ids(db_path=db)) == {"y0", "y1", "y2"}
    top, bottom = vh.performers(limit=1, db_path=db)
    assert [r["subject"] for r in top] == ["S1"]
    assert [r["subject"] for r in bottom] == ["S0"]


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
