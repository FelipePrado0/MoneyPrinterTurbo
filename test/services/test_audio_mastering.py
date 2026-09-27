import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.services import audio_mastering
from app.utils import utils


def _ffmpeg(*args):
    return subprocess.run(
        [utils.get_ffmpeg_binary(), "-hide_banner", "-y", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _integrated_lufs(path):
    out = _ffmpeg("-i", path, "-af", "ebur128", "-f", "null", "-").stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", out)[-1])


def _music_volume(path, start, duration):
    """Mean level of the 1 kHz "music" band only, ignoring the 300 Hz "voice"."""
    out = _ffmpeg(
        "-ss", str(start), "-t", str(duration), "-i", path,
        "-af", "highpass=f=700,highpass=f=700,volumedetect", "-f", "null", "-",
    ).stderr
    return float(re.search(r"mean_volume: (-?[\d.]+) dB", out).group(1))


class AudioMasteringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="audio-mastering-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # Quiet "voice": 6s of speech-band tone, then 4s of silence.
        self.voice = str(self.tmp / "voice.mp3")
        _ffmpeg(
            "-f", "lavfi", "-i", "sine=frequency=300:duration=6:sample_rate=24000",
            "-af", "volume=0.05,apad=pad_dur=4", "-ac", "1", self.voice,
        )
        # Short "music" that has to be looped to cover the whole video.
        self.bgm = str(self.tmp / "bgm.mp3")
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=1000:duration=3", "-ac", "2", self.bgm)
        self.out = str(self.tmp / "master.wav")

    def test_voice_only_is_normalized_to_youtube_loudness(self):
        self.assertTrue(audio_mastering.master(self.voice, None, 0.2, 10.0, self.out))
        self.assertAlmostEqual(_integrated_lufs(self.out), -14.0, delta=1.0)

    def test_music_is_looped_and_ducked_under_speech(self):
        self.assertTrue(audio_mastering.master(self.voice, self.bgm, 0.2, 12.0, self.out))

        duration = float(
            _ffmpeg("-i", self.out, "-f", "null", "-").stderr.rsplit("time=", 1)[1].split()[0].split(":")[-1]
        )
        self.assertGreater(duration, 11.5)
        self.assertAlmostEqual(_integrated_lufs(self.out), -14.0, delta=1.0)
        # Music is pushed down while the voice speaks and comes back after it.
        self.assertLess(_music_volume(self.out, 1.0, 4.0), _music_volume(self.out, 8.0, 3.0) - 3.0)

    def test_returns_false_on_invalid_input(self):
        self.assertFalse(audio_mastering.master(str(self.tmp / "missing.mp3"), None, 0.2, 5.0, self.out))


if __name__ == "__main__":
    unittest.main()
