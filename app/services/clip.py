"""Upload a long video, cut a subtitled vertical clip of at most 60s.

Speech-to-text only (Whisper, via app.services.subtitle) - this feature never
runs TTS. The clip always ends on a sentence boundary when one exists at or
before the 60s cap; otherwise it hard-cuts at the cap. It never exceeds 60s.
"""

import os
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from loguru import logger

from app.models import const
from app.models.schema import VideoAspect, VideoParams
from app.services import material_upload
from app.services import state as sm
from app.services import subtitle
from app.services import video as video_service
from app.utils import utils

MAX_CLIP_SOURCE_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2GB
# Bounds how long a single STT pass can take; not a hard product requirement,
# just a safety valve against multi-hour uploads stalling the task queue.
MAX_CLIP_SOURCE_DURATION_SECONDS = 60 * 60
CLIP_MAX_DURATION_SECONDS = 60.0
MIN_REMAINING_SECONDS = 3.0
# How far past the 60s cap to look for a sentence boundary before giving up
# and hard-cutting. Purely an STT search window, unrelated to the final clip
# length invariant enforced by choose_cut_end().
_CUT_WINDOW_BUFFER_SECONDS = 5.0

_COPY_CHUNK_BYTES = 1024 * 1024
_CLIP_CANVAS_WIDTH, _CLIP_CANVAS_HEIGHT = VideoAspect.portrait.to_resolution()


class ClipValidationError(ValueError):
    """The uploaded source or the requested cut is not valid."""


class ClipServiceError(RuntimeError):
    """The server could not process the clip (STT unavailable, ffmpeg failure, ...)."""


def clip_source_dir(create: bool = True) -> str:
    return utils.storage_dir("clip_sources", create=create)


def validate_source_duration(duration: float) -> None:
    if duration > MAX_CLIP_SOURCE_DURATION_SECONDS:
        max_minutes = MAX_CLIP_SOURCE_DURATION_SECONDS // 60
        raise ClipValidationError(
            f"source video exceeds the {max_minutes} minute limit"
        )


def validate_start_time(
    start_time: float,
    total_duration: float,
    min_remaining: float = MIN_REMAINING_SECONDS,
) -> None:
    if start_time < 0:
        raise ClipValidationError("start_time must not be negative")
    if start_time >= total_duration:
        raise ClipValidationError("start_time is at or past the end of the video")
    if total_duration - start_time < min_remaining:
        raise ClipValidationError(
            f"start_time leaves less than {min_remaining}s of video remaining"
        )


def choose_cut_end(
    segment_ends: list[float],
    available_duration: float,
    max_duration: float = CLIP_MAX_DURATION_SECONDS,
) -> tuple[float, bool]:
    """Pick where the clip should end.

    Never returns more than ``max_duration``. If the source has less than
    ``max_duration`` seconds remaining from the start point, returns that
    natural end (no forced cut, no sentence-boundary requirement). Otherwise
    picks the last sentence boundary at or before ``max_duration``; if none
    exists (e.g. music/silence right up to the cap), hard-cuts at
    ``max_duration`` and reports that fallback via the second return value.
    """
    if available_duration <= max_duration:
        return available_duration, False

    candidates = [end for end in segment_ends if end <= max_duration]
    if candidates:
        return max(candidates), False
    return max_duration, True


def probe_source(path: str) -> tuple[float, bool]:
    """Return ``(duration_seconds, has_audio)`` for an uploaded video."""
    with video_service._open_video_clip_quietly(path, audio=True) as source_clip:
        return float(source_clip.duration), source_clip.audio is not None


def _stage_source_upload(filename: str, source: BinaryIO) -> tuple[str, str]:
    """Chunked, size-capped copy of the upload into the clip sources dir."""
    suffix = Path(filename).suffix.lower()
    target_dir = clip_source_dir(create=True)
    stored_name = f"{uuid4().hex}{suffix}"
    target_path = os.path.join(target_dir, stored_name)

    try:
        source.seek(0)
    except (AttributeError, OSError) as exc:
        raise ClipValidationError("upload is not seekable") from exc

    total_bytes = 0
    try:
        with open(target_path, "wb") as output:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_CLIP_SOURCE_UPLOAD_BYTES:
                    raise ClipValidationError(
                        "source video exceeds the "
                        f"{MAX_CLIP_SOURCE_UPLOAD_BYTES // (1024 * 1024)} MB limit"
                    )
                output.write(chunk)
        if total_bytes == 0:
            raise ClipValidationError("uploaded file is empty")
    except Exception:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise
    return stored_name, target_path


def save_clip_source(filename: str, source: BinaryIO) -> tuple[str, float, bool]:
    """Validate and persist an uploaded clip-source video.

    Returns ``(upload_id, duration_seconds, has_audio)``. ``upload_id`` is
    the stored filename inside ``clip_source_dir()``.
    """
    try:
        safe_name = material_upload.sanitize_material_filename(filename)
    except material_upload.MaterialUploadError as exc:
        raise ClipValidationError(str(exc)) from exc
    if material_upload._material_kind(safe_name) != "video":
        raise ClipValidationError("clip source must be a video file")

    stored_name, target_path = _stage_source_upload(safe_name, source)
    try:
        material_upload._validate_video(target_path)
        duration, has_audio = probe_source(target_path)
        validate_source_duration(duration)
        if not has_audio:
            raise ClipValidationError(
                "uploaded video has no audio track; speech-to-text needs audio"
            )
    except material_upload.MaterialUploadError as exc:
        # e.g. "not a completely decodable video stream" - a rejected upload,
        # same as our own ClipValidationError, not a server-side failure.
        os.remove(target_path)
        raise ClipValidationError(str(exc)) from exc
    except material_upload.MaterialServiceError as exc:
        os.remove(target_path)
        raise ClipServiceError(str(exc)) from exc
    except Exception:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise

    logger.info(
        f"clip source uploaded: upload_id={stored_name}, duration={duration:.1f}s"
    )
    return stored_name, duration, has_audio


def _remove_quietly(path: str) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning(f"failed to remove clip temp file: path={path}, error={exc}")


def _filter_subtitles(subtitles: list[dict], cut_end: float) -> list[dict]:
    """Drop/clamp segments so no subtitle line outlives the trimmed clip."""
    filtered = []
    for item in subtitles:
        if item["start_time"] >= cut_end:
            continue
        filtered.append(
            {
                "msg": item["msg"],
                "start_time": item["start_time"],
                "end_time": min(item["end_time"], cut_end),
            }
        )
    return filtered


def generate_clip(
    task_id: str,
    upload_id: str,
    start_time: float,
    subtitle_position: str,
    subtitle_display_mode: str,
    subtitle_language: str | None = None,
):
    """Task-manager entry point: build the subtitled vertical clip.

    Mirrors ``app.services.task.start`` - always resolves to a terminal task
    state (COMPLETE/FAILED) via ``app.services.state``, never raises.
    """
    sm.state.patch_task(task_id, state=const.TASK_STATE_PROCESSING, progress=0)
    source_path = os.path.join(clip_source_dir(create=True), upload_id)
    task_dir = utils.task_dir(task_id)
    window_audio_path = os.path.join(task_dir, "clip_window_audio.mp3")
    clip_audio_path = os.path.join(task_dir, "clip_audio.mp3")
    clip_video_path = os.path.join(task_dir, "clip_video.mp4")
    subtitle_path = os.path.join(task_dir, "subtitle.srt")
    final_output_path = os.path.join(task_dir, "final.mp4")

    try:
        if not os.path.isfile(source_path):
            raise ClipValidationError("clip source upload not found")

        with video_service._open_video_clip_quietly(
            source_path, audio=True
        ) as source_clip:
            total_duration = float(source_clip.duration)
            validate_start_time(start_time, total_duration)

            window_end = min(
                start_time + CLIP_MAX_DURATION_SECONDS + _CUT_WINDOW_BUFFER_SECONDS,
                total_duration,
            )
            window_clip = source_clip.subclipped(start_time, window_end)
            window_clip.audio.write_audiofile(
                window_audio_path, logger=None
            )
            sm.state.patch_task(task_id, state=const.TASK_STATE_PROCESSING, progress=25)

            subtitles = subtitle.transcribe_segments(
                window_audio_path,
                word_level=(subtitle_display_mode == "word_by_word"),
                language=(subtitle_language or None),
            )
            if subtitles is None:
                raise ClipServiceError(
                    "speech-to-text is unavailable on this server"
                )
            sm.state.patch_task(task_id, state=const.TASK_STATE_PROCESSING, progress=55)

            available_duration = min(
                CLIP_MAX_DURATION_SECONDS, total_duration - start_time
            )
            segment_ends = [item["end_time"] for item in subtitles]
            cut_end, hard_cut_fallback = choose_cut_end(
                segment_ends, available_duration, CLIP_MAX_DURATION_SECONDS
            )

            final_clip = window_clip.subclipped(0, cut_end)
            final_clip.audio.write_audiofile(clip_audio_path, logger=None)

            cropped_clip = video_service._fit_clip_to_canvas(
                final_clip,
                target_width=_CLIP_CANVAS_WIDTH,
                target_height=_CLIP_CANVAS_HEIGHT,
            )
            cropped_clip.without_audio().write_videofile(
                clip_video_path,
                codec="libx264",
                audio=False,
                logger=None,
                fps=source_clip.fps or 30,
            )
        sm.state.patch_task(task_id, state=const.TASK_STATE_PROCESSING, progress=75)

        filtered_subtitles = _filter_subtitles(subtitles, cut_end)
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write(subtitle.subtitles_to_srt(filtered_subtitles))

        params = VideoParams(
            video_subject="",
            video_aspect=VideoAspect.portrait.value,
            subtitle_enabled=True,
            subtitle_position=subtitle_position,
            subtitle_display_mode=subtitle_display_mode,
            voice_volume=1.0,
            # This feature narrates over the uploaded footage's own audio;
            # mixing in random background music would talk over it.
            bgm_type="",
        )
        video_service.generate_video(
            video_path=clip_video_path,
            audio_path=clip_audio_path,
            subtitle_path=subtitle_path,
            output_file=final_output_path,
            params=params,
        )

        sm.state.patch_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            video_file=final_output_path,
            subtitle_file=subtitle_path,
            clip_duration=cut_end,
            hard_cut_fallback=hard_cut_fallback,
        )
        logger.success(
            f"clip generated: task_id={task_id}, duration={cut_end:.1f}s, "
            f"hard_cut_fallback={hard_cut_fallback}"
        )
    except ClipValidationError as exc:
        logger.warning(f"clip generation rejected: task_id={task_id}, error={exc}")
        sm.state.patch_task(
            task_id,
            state=const.TASK_STATE_FAILED,
            failed_stage="validation",
            error=str(exc),
        )
    except Exception as exc:
        logger.exception(f"clip generation failed: task_id={task_id}, error={exc}")
        sm.state.patch_task(
            task_id,
            state=const.TASK_STATE_FAILED,
            failed_stage="pipeline",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        # Source upload and the STT search-window audio are single-use scratch
        # space; only the final clip + subtitle are worth keeping.
        _remove_quietly(source_path)
        _remove_quietly(window_audio_path)
