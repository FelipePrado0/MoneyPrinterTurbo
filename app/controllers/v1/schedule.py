"""CRUD de agendamento de geração de vídeos (feature de scheduling)."""

from datetime import datetime

from fastapi import Depends, Path, Query, Request

from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import CreateScheduleRequest
from app.services import schedule_store
from app.utils import utils

router = new_router(dependencies=[Depends(base.verify_token)])


def _serialize_occurrence(occurrence: dict) -> dict:
    return {
        "id": occurrence["id"],
        "group_id": occurrence["group_id"],
        "generate_at": occurrence["generate_at"].isoformat(),
        "video_subject": occurrence["video_subject"],
        "status": occurrence["status"],
        "task_id": occurrence["task_id"],
        "error": occurrence["error"],
        "created_at": occurrence["created_at"].isoformat(),
    }


@router.post("/schedules", summary="Create a batch of scheduled video occurrences")
def create_schedule(request: Request, body: CreateScheduleRequest):
    request_id = base.get_task_id(request)
    try:
        occurrences = [
            {
                "generate_at": datetime.fromisoformat(item.generate_at),
                "video_subject": item.video_subject,
            }
            for item in body.occurrences
        ]
    except ValueError as exc:
        raise HttpException(
            task_id=request_id,
            status_code=400,
            message=f"{request_id}: invalid generate_at value: {exc}",
        ) from exc

    group_id = schedule_store.create_schedule(
        occurrences=occurrences,
        params=body.params.model_dump(),
    )
    return utils.get_response(200, {"group_id": group_id, "count": len(occurrences)})


@router.get("/schedules", summary="List scheduled occurrences")
def list_schedules(
    request: Request,
    group_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
):
    occurrences = schedule_store.list_occurrences(group_id=group_id, status=status)
    return utils.get_response(200, [_serialize_occurrence(o) for o in occurrences])


@router.delete(
    "/schedules/occurrence/{occurrence_id}",
    summary="Cancel a single pending occurrence",
)
def cancel_occurrence(
    request: Request, occurrence_id: int = Path(..., description="Occurrence ID")
):
    request_id = base.get_task_id(request)
    cancelled = schedule_store.cancel_occurrence(occurrence_id)
    if not cancelled:
        raise HttpException(
            task_id=request_id,
            status_code=404,
            message=f"{request_id}: occurrence not found or already dispatched",
        )
    return utils.get_response(200)


@router.delete("/schedules/{group_id}", summary="Cancel every pending occurrence in a group")
def cancel_schedule_group(
    request: Request, group_id: str = Path(..., description="Schedule group ID")
):
    cancelled_count = schedule_store.cancel_group(group_id)
    return utils.get_response(200, {"cancelled": cancelled_count})
