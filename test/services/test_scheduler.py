from unittest.mock import patch

from app.services import scheduler


def _occurrence(**overrides) -> dict:
    base = {
        "id": 1,
        "group_id": "group-1",
        "generate_at": None,
        "video_subject": "Café todo dia",
        "params": {"video_subject": "Café todo dia", "video_aspect": "9:16"},
        "status": "dispatched",
        "task_id": "task-1",
        "error": None,
        "created_at": None,
    }
    base.update(overrides)
    return base


def test_dispatch_occurrence_starts_task_with_claimed_task_id():
    occurrence = _occurrence()

    with patch.object(scheduler.task_service, "start") as start:
        scheduler._dispatch_occurrence(occurrence)

    assert start.call_count == 1
    task_id, params = start.call_args.args[:2]
    assert task_id == "task-1"
    assert params.video_subject == "Café todo dia"


def test_dispatch_occurrence_uses_youtube_fields_already_in_params():
    """youtube_review_required/publish_offset_hours vem de VideoParams (base
    params da geração), nao de um override proprio do agendamento."""
    occurrence = _occurrence(
        params={
            "video_subject": "Café todo dia",
            "youtube_publish_offset_hours": 3.0,
            "youtube_review_required": True,
        }
    )

    with patch.object(scheduler.task_service, "start") as start:
        scheduler._dispatch_occurrence(occurrence)

    params = start.call_args.args[1]
    assert params.youtube_publish_offset_hours == 3.0
    assert params.youtube_review_required is True


def test_dispatch_occurrence_marks_failed_on_invalid_params_without_starting_task():
    occurrence = _occurrence(params={})  # missing required video_subject

    with (
        patch.object(scheduler.task_service, "start") as start,
        patch.object(scheduler.schedule_store, "mark_failed") as mark_failed,
    ):
        scheduler._dispatch_occurrence(occurrence)

    start.assert_not_called()
    mark_failed.assert_called_once()
    assert mark_failed.call_args.args[0] == 1


def test_dispatch_occurrence_survives_unexpected_start_crash():
    """task.start() ja se blinda sozinho, mas um crash inesperado no dispatcher
    nao pode subir e matar a thread do poller."""
    occurrence = _occurrence()

    with patch.object(scheduler.task_service, "start", side_effect=RuntimeError("boom")):
        scheduler._dispatch_occurrence(occurrence)  # nao deve levantar


def test_poll_once_submits_every_claimed_occurrence():
    claimed = [_occurrence(id=1), _occurrence(id=2)]

    with (
        patch.object(
            scheduler.schedule_store, "claim_due_occurrences", return_value=claimed
        ),
        patch.object(scheduler, "_dispatch_executor") as executor,
    ):
        scheduler._poll_once()

    assert executor.submit.call_count == 2
    dispatched_ids = {call.args[1]["id"] for call in executor.submit.call_args_list}
    assert dispatched_ids == {1, 2}


def test_poll_once_does_not_raise_when_store_fails():
    with patch.object(
        scheduler.schedule_store,
        "claim_due_occurrences",
        side_effect=RuntimeError("db locked"),
    ):
        scheduler._poll_once()  # nao deve levantar, so logar
