import sqlite3
from datetime import datetime, timedelta

import pytest

from app.services import schedule_store

BASE_PARAMS = {"video_subject": "placeholder", "video_aspect": "portrait"}


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "schedule.db")


def _occurrence(when: datetime, subject: str = "Café todo dia"):
    return {"generate_at": when, "video_subject": subject}


def test_create_schedule_persists_one_row_per_occurrence(db_path):
    now = datetime(2026, 3, 5, 9, 0)
    group_id = schedule_store.create_schedule(
        occurrences=[_occurrence(now), _occurrence(now + timedelta(days=1))],
        params=BASE_PARAMS,
        db_path=db_path,
    )

    rows = schedule_store.list_occurrences(db_path=db_path)
    assert len(rows) == 2
    assert all(row["group_id"] == group_id for row in rows)
    assert {row["status"] for row in rows} == {"pending"}


def test_create_schedule_stores_per_occurrence_subject(db_path):
    now = datetime(2026, 3, 5, 9, 0)
    schedule_store.create_schedule(
        occurrences=[
            _occurrence(now, "Assunto A"),
            _occurrence(now + timedelta(days=1), "Assunto B"),
        ],
        params=BASE_PARAMS,
        db_path=db_path,
    )

    rows = sorted(
        schedule_store.list_occurrences(db_path=db_path),
        key=lambda r: r["generate_at"],
    )
    assert [row["video_subject"] for row in rows] == ["Assunto A", "Assunto B"]


def test_claim_due_occurrences_only_claims_past_pending_rows(db_path):
    past = datetime(2026, 3, 5, 9, 0)
    future = datetime(2026, 3, 20, 9, 0)
    schedule_store.create_schedule(
        occurrences=[_occurrence(past), _occurrence(future)],
        params=BASE_PARAMS,
        db_path=db_path,
    )

    claimed = schedule_store.claim_due_occurrences(
        now=datetime(2026, 3, 6), task_id_factory=lambda: "task-1", db_path=db_path
    )
    assert len(claimed) == 1
    assert claimed[0]["video_subject"] == "Café todo dia"
    assert claimed[0]["status"] == "dispatched"
    assert claimed[0]["task_id"] == "task-1"

    row = schedule_store.list_occurrences(db_path=db_path, status="pending")
    assert len(row) == 1
    assert row[0]["video_subject"] == "Café todo dia"  # a futura, ainda pending


def test_claim_due_occurrences_never_claims_the_same_row_twice(db_path):
    """Simula dois poll ticks (ou dois processos) disputando a mesma
    ocorrencia due: so o primeiro claim pode vencer."""
    past = datetime(2026, 3, 5, 9, 0)
    schedule_store.create_schedule(
        occurrences=[_occurrence(past)], params=BASE_PARAMS, db_path=db_path
    )

    first = schedule_store.claim_due_occurrences(
        now=datetime(2026, 3, 6), task_id_factory=lambda: "task-1", db_path=db_path
    )
    second = schedule_store.claim_due_occurrences(
        now=datetime(2026, 3, 6), task_id_factory=lambda: "task-2", db_path=db_path
    )

    assert len(first) == 1
    assert second == []

    row = schedule_store.list_occurrences(db_path=db_path)[0]
    assert row["status"] == "dispatched"
    assert row["task_id"] == "task-1"


def test_mark_failed_records_error_and_status(db_path):
    schedule_store.create_schedule(
        occurrences=[_occurrence(datetime(2026, 3, 5, 9, 0))],
        params=BASE_PARAMS,
        db_path=db_path,
    )
    occurrence_id = schedule_store.list_occurrences(db_path=db_path)[0]["id"]

    schedule_store.mark_failed(occurrence_id, error="ffmpeg missing", db_path=db_path)

    row = schedule_store.list_occurrences(db_path=db_path)[0]
    assert row["status"] == "failed"
    assert row["error"] == "ffmpeg missing"


def test_cancel_occurrence_marks_single_row_cancelled(db_path):
    group_id = schedule_store.create_schedule(
        occurrences=[
            _occurrence(datetime(2026, 3, 5, 9, 0)),
            _occurrence(datetime(2026, 3, 6, 9, 0)),
        ],
        params=BASE_PARAMS,
        db_path=db_path,
    )
    rows = schedule_store.list_occurrences(group_id=group_id, db_path=db_path)

    schedule_store.cancel_occurrence(rows[0]["id"], db_path=db_path)

    updated = schedule_store.list_occurrences(group_id=group_id, db_path=db_path)
    statuses = {row["id"]: row["status"] for row in updated}
    assert statuses[rows[0]["id"]] == "cancelled"
    assert statuses[rows[1]["id"]] == "pending"


def test_cancel_group_marks_all_pending_rows_cancelled(db_path):
    group_id = schedule_store.create_schedule(
        occurrences=[
            _occurrence(datetime(2026, 3, 5, 9, 0)),
            _occurrence(datetime(2026, 3, 6, 9, 0)),
        ],
        params=BASE_PARAMS,
        db_path=db_path,
    )

    cancelled_count = schedule_store.cancel_group(group_id, db_path=db_path)

    assert cancelled_count == 2
    rows = schedule_store.list_occurrences(group_id=group_id, db_path=db_path)
    assert all(row["status"] == "cancelled" for row in rows)


def test_cancel_group_does_not_touch_dispatched_rows(db_path):
    group_id = schedule_store.create_schedule(
        occurrences=[_occurrence(datetime(2026, 3, 5, 9, 0))],
        params=BASE_PARAMS,
        db_path=db_path,
    )
    occurrence_id = schedule_store.list_occurrences(group_id=group_id, db_path=db_path)[
        0
    ]["id"]
    schedule_store.mark_dispatched(occurrence_id, task_id="task-1", db_path=db_path)

    cancelled_count = schedule_store.cancel_group(group_id, db_path=db_path)

    assert cancelled_count == 0
    row = schedule_store.list_occurrences(group_id=group_id, db_path=db_path)[0]
    assert row["status"] == "dispatched"


def test_list_occurrences_filters_by_status(db_path):
    group_id = schedule_store.create_schedule(
        occurrences=[
            _occurrence(datetime(2026, 3, 5, 9, 0)),
            _occurrence(datetime(2026, 3, 6, 9, 0)),
        ],
        params=BASE_PARAMS,
        db_path=db_path,
    )
    occurrence_id = schedule_store.list_occurrences(group_id=group_id, db_path=db_path)[
        0
    ]["id"]
    schedule_store.mark_dispatched(occurrence_id, task_id="task-1", db_path=db_path)

    pending = schedule_store.list_occurrences(status="pending", db_path=db_path)
    assert len(pending) == 1


def test_params_round_trips_through_json(db_path):
    params = {"video_subject": "x", "voice_name": "pt-BR-AntonioNeural", "video_count": 2}
    schedule_store.create_schedule(
        occurrences=[_occurrence(datetime(2026, 3, 5, 9, 0))],
        params=params,
        db_path=db_path,
    )
    row = schedule_store.list_occurrences(db_path=db_path)[0]
    assert row["params"]["voice_name"] == "pt-BR-AntonioNeural"
    assert row["params"]["video_count"] == 2


def test_creates_table_on_first_connect(db_path):
    schedule_store.create_schedule(
        occurrences=[_occurrence(datetime(2026, 3, 5, 9, 0))],
        params=BASE_PARAMS,
        db_path=db_path,
    )
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "schedule_occurrences" in tables


class TestUpdateVideoSubject:
    def test_updates_the_column_and_the_embedded_params(self, db_path):
        group_id = schedule_store.create_schedule(
            occurrences=[_occurrence(datetime(2026, 3, 5, 9, 0), "__AUTO__")],
            params=BASE_PARAMS,
            db_path=db_path,
        )
        occurrence_id = schedule_store.list_occurrences(
            group_id=group_id, db_path=db_path
        )[0]["id"]

        schedule_store.update_video_subject(
            occurrence_id, "cute puppies playing", db_path=db_path
        )

        row = schedule_store.list_occurrences(group_id=group_id, db_path=db_path)[0]
        assert row["video_subject"] == "cute puppies playing"
        assert row["params"]["video_subject"] == "cute puppies playing"

    def test_unknown_occurrence_id_is_a_silent_no_op(self, db_path):
        schedule_store.create_schedule(
            occurrences=[_occurrence(datetime(2026, 3, 5, 9, 0))],
            params=BASE_PARAMS,
            db_path=db_path,
        )
        # Must not raise even if the row doesn't exist.
        schedule_store.update_video_subject(9999, "whatever", db_path=db_path)


class TestListRecentResolvedTopics:
    def test_returns_dispatched_subjects_most_recent_first(self, db_path):
        schedule_store.create_schedule(
            occurrences=[
                _occurrence(datetime(2026, 3, 5, 9, 0), "topic A"),
                _occurrence(datetime(2026, 3, 6, 9, 0), "topic B"),
            ],
            params=BASE_PARAMS,
            db_path=db_path,
        )
        for occurrence in schedule_store.list_occurrences(db_path=db_path):
            schedule_store.mark_dispatched(occurrence["id"], "task-x", db_path=db_path)

        topics = schedule_store.list_recent_resolved_topics(db_path=db_path)
        assert topics == ["topic B", "topic A"]

    def test_excludes_pending_and_sentinel_rows(self, db_path):
        schedule_store.create_schedule(
            occurrences=[
                _occurrence(datetime(2026, 3, 5, 9, 0), "__AUTO_TREND_TOPIC__"),
                _occurrence(datetime(2026, 3, 6, 9, 0), "resolved topic"),
            ],
            params=BASE_PARAMS,
            db_path=db_path,
        )
        rows = schedule_store.list_occurrences(db_path=db_path)
        # Only the second (resolved) row gets dispatched+resolved; the first
        # stays pending, still holding the sentinel.
        resolved_row = next(r for r in rows if r["video_subject"] == "resolved topic")
        schedule_store.mark_dispatched(resolved_row["id"], "task-x", db_path=db_path)

        topics = schedule_store.list_recent_resolved_topics(db_path=db_path)
        assert topics == ["resolved topic"]

    def test_respects_limit(self, db_path):
        schedule_store.create_schedule(
            occurrences=[
                _occurrence(datetime(2026, 3, day, 9, 0), f"topic {day}")
                for day in range(5, 10)
            ],
            params=BASE_PARAMS,
            db_path=db_path,
        )
        for occurrence in schedule_store.list_occurrences(db_path=db_path):
            schedule_store.mark_dispatched(occurrence["id"], "task-x", db_path=db_path)

        topics = schedule_store.list_recent_resolved_topics(limit=2, db_path=db_path)
        assert topics == ["topic 9", "topic 8"]
