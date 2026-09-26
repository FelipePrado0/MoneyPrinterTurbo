"""ModelBest VoxCPM TTS provider (split out of ``voice.py``)."""

import base64
import io
import json
import math
import os
import tempfile
import time
from typing import Union

import requests
from edge_tts import SubMaker
from loguru import logger
from moviepy.audio.io.AudioFileClip import AudioFileClip

from app.config import config
from app.services.voice_common import (
    _configure_pydub_ffmpeg,
    ensure_file_path_exists,
    ensure_legacy_submaker_fields,
    populate_legacy_submaker_with_full_text,
)
from app.utils import utils

VOXCPM_DEFAULT_BASE_URL = "https://api.modelbest.cn/v1"
VOXCPM_DEFAULT_VOICE = "default"
_VOXCPM_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}
_VOXCPM_RETRY_DELAY_SECONDS = (1.0, 2.0)


def get_voxcpm_voices(voice_id: str | None = None) -> list[str]:
    """Return the ModelBest VoxCPM voice selected in the local configuration.

    ModelBest accepts ``default`` when no explicit voice is selected. Voice
    design is expressed in the input text and voice cloning requires a separate
    reference-audio workflow, so the first integration deliberately keeps the
    standard TTS selector to one configured voice id.
    """
    voice_id = str(
        voice_id
        or config.voxcpm.get("voice_id", VOXCPM_DEFAULT_VOICE)
        or VOXCPM_DEFAULT_VOICE
    ).strip()
    return [f"voxcpm:{voice_id}"]


def is_voxcpm_voice(voice_name: str | None) -> bool:
    return (voice_name or "").startswith("voxcpm:")


def _iter_voxcpm_sse_events(response):
    """Yield JSON payloads from ModelBest's Server-Sent Event stream."""
    event_data = []
    for raw_line in response.iter_lines(decode_unicode=True):
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line:
            if event_data:
                try:
                    yield json.loads("\n".join(event_data))
                except (TypeError, ValueError) as exc:
                    raise ValueError("VoxCPM returned invalid SSE event data") from exc
                event_data = []
            continue
        if line.startswith("data:"):
            event_data.append(line.removeprefix("data:").strip())
    if event_data:
        try:
            yield json.loads("\n".join(event_data))
        except (TypeError, ValueError) as exc:
            raise ValueError("VoxCPM returned invalid trailing SSE event data") from exc


def voxcpm_tts(
    text: str,
    voice_id: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """Generate speech through ModelBest's VoxCPM Audio Speech API.

    ModelBest always streams Base64-encoded WAV chunks through SSE. The
    assembled WAV is decoded and exported to the project's requested output
    format so that the regular subtitle and video paths remain unchanged.
    ModelBest does not define a numeric speed field, so ``voice_rate`` is not
    sent. ``voice_volume`` is applied later by MoneyPrinterTurbo's video mixer.
    """
    from pydub import AudioSegment

    text = (text or "").strip()
    if not text:
        logger.error("VoxCPM TTS text is empty")
        return None

    api_key = str(config.voxcpm.get("api_key", "") or "").strip()
    if not api_key:
        logger.error("VoxCPM API key is not set")
        return None

    base_url = str(
        config.voxcpm.get("base_url", VOXCPM_DEFAULT_BASE_URL)
        or VOXCPM_DEFAULT_BASE_URL
    ).strip().rstrip("/")
    model_id = str(config.voxcpm.get("model_id", "") or "").strip()
    if not model_id:
        logger.error("VoxCPM model ID is not set")
        return None
    voice_id = str(voice_id or VOXCPM_DEFAULT_VOICE).strip() or VOXCPM_DEFAULT_VOICE

    url = f"{base_url}/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model_id,
        "input": text,
        "voice": voice_id,
        "response_format": "wav",
        "stream": True,
    }
    _configure_pydub_ffmpeg(AudioSegment)

    for attempt in range(3):
        temporary_audio = None
        response = None
        try:
            logger.info(
                f"start VoxCPM TTS, model: {model_id}, voice: {voice_id}, "
                f"try: {attempt + 1}"
            )
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=(10, 120),
            )
            if response.status_code != 200:
                logger.error(
                    f"VoxCPM TTS failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                if response.status_code in _VOXCPM_NON_RETRYABLE_STATUS_CODES:
                    return None
                if attempt < 2:
                    time.sleep(_VOXCPM_RETRY_DELAY_SECONDS[attempt])
                continue

            audio_chunks = []
            completed = False
            for event in _iter_voxcpm_sse_events(response):
                event_type = event.get("type")
                if event_type == "speech.audio.delta":
                    encoded_chunk = event.get("audio")
                    if not isinstance(encoded_chunk, str) or not encoded_chunk:
                        raise ValueError("VoxCPM returned an empty audio chunk")
                    try:
                        audio_chunks.append(base64.b64decode(encoded_chunk, validate=True))
                    except (ValueError, TypeError) as exc:
                        raise ValueError("VoxCPM returned invalid Base64 audio") from exc
                elif event_type == "speech.audio.done":
                    completed = True
                    break

            if not completed:
                raise ValueError("VoxCPM stream ended before speech.audio.done")
            audio_bytes = b"".join(audio_chunks)
            if not audio_bytes:
                raise ValueError("VoxCPM returned no audio data")

            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
            if len(audio_segment) <= 0:
                raise ValueError("VoxCPM returned an empty WAV")

            ensure_file_path_exists(voice_file)
            output_format = utils.parse_extension(voice_file) or "mp3"
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(os.path.abspath(voice_file)),
                suffix=f".{output_format}",
                delete=False,
            ) as output:
                temporary_audio = output.name
            audio_segment.export(temporary_audio, format=output_format)

            audio_clip = AudioFileClip(temporary_audio)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()
            if not math.isfinite(audio_duration) or audio_duration <= 0:
                raise ValueError("VoxCPM produced an invalid audio duration")

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            os.replace(temporary_audio, voice_file)
            logger.success(f"VoxCPM TTS succeeded: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except requests.RequestException as exc:
            logger.error(f"VoxCPM TTS request failed: {exc}")
            if attempt < 2:
                time.sleep(_VOXCPM_RETRY_DELAY_SECONDS[attempt])
        except Exception as exc:
            # Invalid SSE/WAV data and local conversion failures are deterministic;
            # retrying the same response cannot repair them.
            logger.error(f"VoxCPM TTS failed: {exc}")
            return None
        finally:
            close_response = getattr(response, "close", None)
            if callable(close_response):
                close_response()
            if temporary_audio and os.path.exists(temporary_audio):
                try:
                    os.unlink(temporary_audio)
                except OSError as exc:
                    logger.warning(f"could not remove temporary VoxCPM audio: {exc}")

    return None
