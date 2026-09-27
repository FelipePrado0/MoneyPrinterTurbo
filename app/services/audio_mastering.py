"""Final audio mix: treated voice, music ducked under speech, YouTube loudness."""

import os
import subprocess

from loguru import logger

from app.utils import utils

# YouTube normalizes playback to about -14 LUFS; mastering to it avoids both
# being turned down and sounding quieter than the next Short.
LOUDNESS = "loudnorm=I=-14:TP=-1.5:LRA=11"
VOICE_CHAIN = (
    "highpass=f=80,"
    "acompressor=threshold=-20dB:ratio=3:attack=5:release=120:makeup=2,"
    "loudnorm=I=-16:TP=-2:LRA=11"
)
# Threshold is linear amplitude (~-34 dBFS): any speech pushes the music down.
DUCKING = "sidechaincompress=threshold=0.02:ratio=10:attack=20:release=500"
FORMAT = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
FFMPEG_TIMEOUT_SECONDS = 300


def master(
    voice_file: str,
    bgm_file: str | None,
    bgm_volume: float,
    duration: float,
    output_file: str,
) -> bool:
    """Write the mastered track (``duration`` seconds, 48 kHz wav) to ``output_file``.

    The music is looped to cover the whole video and fades out in the last 3s.
    Returns False on any failure so the caller can keep its plain mix.
    """
    duration = max(float(duration), 0.1)
    voice = f"[0:a]{FORMAT},{VOICE_CHAIN},{FORMAT},apad=whole_dur={duration:.3f}"
    inputs = ["-i", voice_file]
    if bgm_file:
        inputs += ["-stream_loop", "-1", "-i", bgm_file]
        fade_start = max(duration - 3, 0)
        graph = (
            f"{voice},asplit=2[v][sc];"
            f"[1:a]{FORMAT},volume={bgm_volume},atrim=0:{duration:.3f},"
            f"afade=t=out:st={fade_start:.3f}:d=3[b];"
            f"[b][sc]{DUCKING}[bd];"
            f"[v][bd]amix=inputs=2:duration=first:normalize=0,{LOUDNESS},{FORMAT}[out]"
        )
    else:
        graph = f"{voice},{LOUDNESS},{FORMAT}[out]"

    cmd = [
        utils.get_ffmpeg_binary(), "-hide_banner", "-y", *inputs,
        "-filter_complex", graph, "-map", "[out]",
        "-t", f"{duration:.3f}", "-c:a", "pcm_s16le", output_file,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=FFMPEG_TIMEOUT_SECONDS
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"audio mastering failed to run: {exc}")
        return False
    if result.returncode != 0 or not os.path.isfile(output_file):
        logger.warning(f"audio mastering failed: {(result.stderr or '').strip()[-500:]}")
        return False
    return True
