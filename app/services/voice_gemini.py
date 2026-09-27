"""Google Gemini TTS provider."""

import base64
import io
from typing import Union

from edge_tts import SubMaker
from loguru import logger

from app.config import config
from app.services.voice_common import (
    _configure_pydub_ffmpeg,
    ensure_file_path_exists,
    ensure_legacy_submaker_fields,
    populate_legacy_submaker_with_full_text,
)

GEMINI_TTS_VOICES = (
    ("Zephyr", "Bright"),
    ("Puck", "Upbeat"),
    ("Charon", "Informative"),
    ("Kore", "Firm"),
    ("Fenrir", "Excitable"),
    ("Leda", "Youthful"),
    ("Orus", "Firm"),
    ("Aoede", "Breezy"),
    ("Callirrhoe", "Easy-going"),
    ("Autonoe", "Bright"),
    ("Enceladus", "Breathy"),
    ("Iapetus", "Clear"),
    ("Umbriel", "Easy-going"),
    ("Algieba", "Smooth"),
    ("Despina", "Smooth"),
    ("Erinome", "Clear"),
    ("Algenib", "Gravelly"),
    ("Rasalgethi", "Informative"),
    ("Laomedeia", "Upbeat"),
    ("Achernar", "Soft"),
    ("Alnilam", "Firm"),
    ("Schedar", "Even"),
    ("Gacrux", "Mature"),
    ("Pulcherrima", "Forward"),
    ("Achird", "Friendly"),
    ("Zubenelgenubi", "Casual"),
    ("Vindemiatrix", "Gentle"),
    ("Sadachbia", "Lively"),
    ("Sadaltager", "Knowledgeable"),
    ("Sulafat", "Warm"),
)


def get_gemini_voices() -> list[str]:
    """
    获取 Gemini TTS 官方预置音色列表。

    Google 没有为这些音色发布性别元数据，因此下拉框使用官方风格描述，
    避免把推测的性别写进持久化 voice id。音色目录来源：
    https://ai.google.dev/gemini-api/docs/speech-generation#voice-options

    Returns:
        声音列表，格式为 ["gemini:Zephyr-Bright", "gemini:Puck-Upbeat", ...]
    """
    return [f"gemini:{voice}-{style}" for voice, style in GEMINI_TTS_VOICES]


def is_gemini_voice(voice_name: str):
    """检查是否是Gemini TTS的声音"""
    return voice_name.startswith("gemini:")


def parse_gemini_voice_name(voice_name: str | None) -> str:
    """从新旧 Gemini 下拉框值中提取 Google API 使用的预置音色名称。"""
    if not is_gemini_voice(voice_name or ""):
        return ""
    return (voice_name or "").split(":", 1)[1].split("-", 1)[0].strip()


DEFAULT_TTS_MODEL = "gemini-3.1-flash-tts-preview"
DEFAULT_FALLBACK_VOICES = ("Puck", "Zephyr")
REQUEST_TIMEOUT_MS = 120_000


def api_keys() -> list[str]:
    """Keys in fallback order: ``gemini_api_keys`` first, then the legacy single key."""
    keys = config.app.get("gemini_api_keys") or []
    if isinstance(keys, str):
        keys = keys.split(",")
    keys = [key.strip() for key in keys if key and key.strip()]
    legacy = (config.app.get("gemini_api_key") or "").strip()
    if legacy and legacy not in keys:
        keys.append(legacy)
    return keys


def _voice_chain(voice_name: str) -> list[str]:
    fallbacks = config.app.get("gemini_tts_fallback_voices", DEFAULT_FALLBACK_VOICES)
    return list(dict.fromkeys(v for v in [voice_name, *fallbacks] if v))


def _request_pcm(api_key: str, model: str, contents: str, voice_name: str) -> bytes:
    from google import genai
    from google.genai import types

    generation_config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )
    with genai.Client(
        api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
    ) as client:
        response = client.models.generate_content(
            model=model, contents=contents, config=generation_config
        )

    for candidate in response.candidates or []:
        for part in getattr(candidate.content, "parts", None) or []:
            data = getattr(getattr(part, "inline_data", None), "data", None)
            if data:
                return base64.b64decode(data) if isinstance(data, str) else data
    raise RuntimeError("no audio data in Gemini response")


def gemini_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Gemini TTS with fallback: each voice of the chain (requested voice, then
    ``gemini_tts_fallback_voices``) is tried with every key in order before
    moving to the next voice. Pace comes from ``gemini_tts_style``, so
    ``voice_rate`` and ``voice_volume`` are ignored.

    Returns a SubMaker with estimated sentence timing, flagged with
    ``needs_alignment`` so the subtitle step can realign it with Whisper, and
    ``tts_voice`` naming the voice that actually spoke. None if all failed.
    """
    from pydub import AudioSegment

    _configure_pydub_ffmpeg(AudioSegment)

    keys = api_keys()
    if not keys:
        logger.error("Gemini API key is not set")
        return None

    model = config.app.get("gemini_tts_model") or DEFAULT_TTS_MODEL
    style = (config.app.get("gemini_tts_style") or "").strip()
    contents = f"{style} {text}" if style else text

    for voice in _voice_chain(voice_name):
        for index, key in enumerate(keys, start=1):
            try:
                pcm = _request_pcm(key, model, contents, voice)
                # Gemini returns 16-bit mono PCM at 24 kHz.
                audio_segment = AudioSegment.from_file(
                    io.BytesIO(pcm), format="raw", frame_rate=24000, channels=1, sample_width=2
                )
                ensure_file_path_exists(voice_file)
                audio_segment.export(voice_file, format="mp3").close()
            except Exception as exc:
                error = str(exc)
                for secret in keys:
                    error = error.replace(secret, "***")
                logger.warning(f"Gemini TTS failed, voice: {voice}, key#{index}: {error[:300]}")
                continue

            logger.info(f"Gemini TTS completed, voice: {voice}, key#{index}, file: {voice_file}")
            sub_maker = populate_legacy_submaker_with_full_text(
                sub_maker=ensure_legacy_submaker_fields(SubMaker()),
                text=text,
                audio_duration_seconds=len(audio_segment) / 1000.0,
            )
            sub_maker.tts_voice = f"gemini:{voice}"
            sub_maker.needs_alignment = True
            return sub_maker

    logger.error("Gemini TTS failed for every voice and key")
    return None
