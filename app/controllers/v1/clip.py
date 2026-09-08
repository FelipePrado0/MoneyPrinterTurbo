from fastapi import Depends, Path, Request, UploadFile
from fastapi.params import File
from loguru import logger

from app.config import config
from app.controllers import base
from app.controllers.manager.base_manager import TaskQueueFullError
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.controllers.manager.redis_manager import RedisTaskManager
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import (
    ClipGenerateRequest,
    ClipTaskQueryResponse,
    ClipTaskResponse,
    ClipUploadResponse,
)
from app.services import clip as clip_service
from app.services import state as sm
from app.utils import utils

router = new_router(dependencies=[Depends(base.verify_token)])

_enable_redis = config.app.get("enable_redis", False)
_redis_host = config.app.get("redis_host", "localhost")
_redis_port = config.app.get("redis_port", 6379)
_redis_db = config.app.get("redis_db", 0)
_redis_password = config.app.get("redis_password", None)
_max_concurrent_tasks = config.app.get("max_concurrent_tasks", 5)
_max_queued_tasks = config.app.get("max_queued_tasks", 100)


def _build_redis_url(host: str, port: int, db: int, password: str | None) -> str:
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/{db}"


if _enable_redis:
    task_manager = RedisTaskManager(
        max_concurrent_tasks=_max_concurrent_tasks,
        redis_url=_build_redis_url(_redis_host, _redis_port, _redis_db, _redis_password),
        max_queued_tasks=_max_queued_tasks,
    )
else:
    task_manager = InMemoryTaskManager(
        max_concurrent_tasks=_max_concurrent_tasks,
        max_queued_tasks=_max_queued_tasks,
    )


@router.post(
    "/clip_uploads",
    response_model=ClipUploadResponse,
    summary="Upload a long source video for the clip-from-video feature",
)
def upload_clip_source(request: Request, file: UploadFile = File(...)):
    request_id = base.get_task_id(request)
    try:
        upload_id, duration, has_audio = clip_service.save_clip_source(
            file.filename, file.file
        )
    except clip_service.ClipValidationError as exc:
        logger.warning(
            f"clip source upload rejected: request_id={request_id}, error={exc}"
        )
        raise HttpException(
            task_id=request_id, status_code=400, message=f"{request_id}: {exc}"
        )
    except Exception as exc:
        logger.error(
            f"clip source upload failed: request_id={request_id}, error={exc}"
        )
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"{request_id}: clip source validation is unavailable",
        )

    return utils.get_response(
        200, {"upload_id": upload_id, "duration": duration, "has_audio": has_audio}
    )


@router.post(
    "/clip_tasks",
    response_model=ClipTaskResponse,
    summary="Generate a subtitled vertical clip (<=60s) from an uploaded video",
)
def create_clip_task(request: Request, body: ClipGenerateRequest):
    request_id = base.get_task_id(request)
    task_id = utils.get_uuid()
    try:
        sm.state.update_task(task_id, kind="clip")
        try:
            task_manager.add_task(
                clip_service.generate_clip,
                task_id=task_id,
                upload_id=body.upload_id,
                start_time=body.start_time,
                subtitle_position=body.subtitle_position,
                subtitle_display_mode=body.subtitle_display_mode,
                subtitle_language=body.subtitle_language,
            )
        except Exception:
            sm.state.delete_task(task_id)
            raise
        return utils.get_response(200, {"task_id": task_id})
    except TaskQueueFullError as exc:
        raise HttpException(
            task_id=task_id, status_code=429, message=f"{request_id}: {exc}"
        )


@router.get(
    "/clip_tasks/{task_id}",
    response_model=ClipTaskQueryResponse,
    summary="Query a clip generation task's status",
)
def get_clip_task(request: Request, task_id: str = Path(..., description="Task ID")):
    request_id = base.get_task_id(request)
    task = sm.state.get_task(task_id)
    if task and task.get("kind") == "clip":
        return utils.get_response(200, task)

    raise HttpException(
        task_id=task_id, status_code=404, message=f"{request_id}: task not found"
    )
