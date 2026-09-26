"""Autopilot control, video history and OpenRouter free-model status."""

from typing import Literal

from fastapi import Depends, Query, Request

from app.config import config
from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import AutopilotSettingsUpdate
from app.services import autopilot, llm_free_models, video_history
from app.utils import utils

router = new_router(dependencies=[Depends(base.verify_token)])

VideoStatus = Literal[
    "generating", "generated", "published", "failed", "rejected"
]


@router.get("/autopilot/status", summary="Autopilot state, today's slots and quota")
def get_autopilot_status(request: Request):
    return utils.get_response(200, autopilot.status())


@router.get("/autopilot/config", summary="Current autopilot settings")
def get_autopilot_config(request: Request):
    return utils.get_response(200, autopilot.load_settings().model_dump())


@router.put("/autopilot/config", summary="Update autopilot settings (partial)")
def update_autopilot_config(request: Request, body: AutopilotSettingsUpdate):
    try:
        settings = autopilot.save_settings(body.model_dump(exclude_none=True))
    except ValueError as exc:
        request_id = base.get_task_id(request)
        raise HttpException(
            task_id=request_id,
            status_code=400,
            message=f"{request_id}: invalid autopilot settings: {exc}",
        ) from exc
    return utils.get_response(200, settings.model_dump())


@router.post("/autopilot/pause", summary="Pause the autopilot and cancel today's pending slots")
def pause_autopilot(request: Request):
    autopilot.pause()
    return utils.get_response(200)


@router.post("/autopilot/resume", summary="Resume the autopilot and re-plan today")
def resume_autopilot(request: Request):
    autopilot.resume()
    return utils.get_response(200)


@router.get("/videos", summary="Generated video history, newest first")
def list_video_history(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status: VideoStatus | None = Query(default=None),
):
    items, total = video_history.list_videos(limit=limit, offset=offset, status=status)
    return utils.get_response(200, {"items": items, "total": total})


@router.get("/llm/free-models", summary="OpenRouter models in fallback order")
def list_free_models(request: Request):
    return utils.get_response(
        200,
        llm_free_models.status(
            llm_free_models.preferred_model(config.app),
            bool(config.app.get("llm_free_only", True)),
        ),
    )
