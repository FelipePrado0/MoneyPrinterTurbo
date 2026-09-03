"""Background dispatcher: turns due schedule occurrences into real tasks.

Runs a daemon polling thread started from ``app/asgi.py``'s lifespan. Every
tick reads due rows from ``schedule_store`` (sqlite, survives restarts) and
hands each to a small thread pool so a slow/blocked ``task.start()`` never
delays the next poll or the other occurrences in the same batch.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from uuid import uuid4

from loguru import logger

from app.models.schema import VideoParams
from app.services import schedule_store
from app.services import task as task_service

POLL_INTERVAL_SECONDS = 30.0

_dispatch_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mpt-schedule-dispatch",
)
_poll_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _dispatch_occurrence(occurrence: dict) -> None:
    """Build VideoParams from an already-claimed occurrence and run the pipeline.

    The occurrence was already flipped to ``dispatched`` atomically in
    ``schedule_store.claim_due_occurrences``, before this ever reaches the
    executor: no risk of a later poll tick re-claiming and re-submitting it.
    YouTube publish overrides (title/description/tags/offset/review) are
    whatever the base ``VideoParams`` already carried at schedule-creation
    time, same as any other generation; nothing schedule-specific to apply
    here.
    """
    occurrence_id = occurrence["id"]
    task_id = occurrence["task_id"]
    try:
        params = VideoParams(**occurrence["params"])
    except Exception as exc:
        logger.exception(
            f"invalid schedule occurrence params, skipping: "
            f"occurrence_id={occurrence_id}, error={exc}"
        )
        schedule_store.mark_failed(occurrence_id, str(exc))
        return

    logger.info(
        f"dispatching scheduled video: occurrence_id={occurrence_id}, "
        f"task_id={task_id}, subject={params.video_subject!r}"
    )
    try:
        task_service.start(task_id, params)
    except Exception as exc:
        # task.start() 已经把所有已知失败转成任务失败状态；这里只是防止
        # 未预期的崩溃杀死调度线程池里的这个 worker。
        logger.exception(
            f"scheduled task crashed unexpectedly: task_id={task_id}, error={exc}"
        )


def _poll_once() -> None:
    try:
        claimed = schedule_store.claim_due_occurrences(
            now=datetime.now(), task_id_factory=lambda: uuid4().hex
        )
    except Exception as exc:
        logger.exception(f"failed to poll schedule occurrences: {exc}")
        return

    for occurrence in claimed:
        _dispatch_executor.submit(_dispatch_occurrence, occurrence)


def _poll_loop() -> None:
    while not _stop_event.is_set():
        _poll_once()
        _stop_event.wait(POLL_INTERVAL_SECONDS)


def start_scheduler() -> None:
    """Start the background polling thread. Safe to call more than once."""
    global _poll_thread
    if _poll_thread is not None and _poll_thread.is_alive():
        return
    _stop_event.clear()
    _poll_thread = threading.Thread(
        target=_poll_loop, daemon=True, name="mpt-schedule-poll"
    )
    _poll_thread.start()
    logger.info("video schedule poller started")


def stop_scheduler() -> None:
    _stop_event.set()
