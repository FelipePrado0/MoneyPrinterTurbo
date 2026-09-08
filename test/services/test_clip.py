"""Tests for app.services.clip: the "upload a long video, cut a subtitled
vertical clip of at most 60s, ending on a sentence boundary" feature.

The cut-selection logic (``choose_cut_end``) is pure business logic with a
hard invariant ("never exceed 60s") and several non-trivial branches, so it
is tested here before/independently of any real ffmpeg/whisper I/O, per the
project's TDD rule for non-trivial branching logic.
"""

import io
import unittest
from unittest import mock

from app.services import clip
from app.services import material_upload


class TestChooseCutEnd(unittest.TestCase):
    def test_picks_last_sentence_boundary_before_max_duration(self):
        # Sentence endings at 12.5s, 38.0s, 59.9s, and 61.2s (past the cap).
        segment_ends = [12.5, 38.0, 59.9, 61.2]
        cut_end, hard_cut = clip.choose_cut_end(
            segment_ends, available_duration=65.0, max_duration=60.0
        )
        self.assertEqual(cut_end, 59.9)
        self.assertFalse(hard_cut)

    def test_never_exceeds_max_duration_even_with_boundary_right_at_it(self):
        segment_ends = [10.0, 60.0, 60.0001, 90.0]
        cut_end, hard_cut = clip.choose_cut_end(
            segment_ends, available_duration=90.0, max_duration=60.0
        )
        self.assertLessEqual(cut_end, 60.0)
        self.assertEqual(cut_end, 60.0)
        self.assertFalse(hard_cut)

    def test_hard_cuts_at_max_duration_when_no_boundary_found(self):
        # e.g. background music/silence for the whole window, no speech end
        # detected before the cap.
        segment_ends = [61.0, 70.0]
        cut_end, hard_cut = clip.choose_cut_end(
            segment_ends, available_duration=90.0, max_duration=60.0
        )
        self.assertEqual(cut_end, 60.0)
        self.assertTrue(hard_cut)

    def test_hard_cuts_at_max_duration_when_no_speech_at_all(self):
        cut_end, hard_cut = clip.choose_cut_end(
            [], available_duration=90.0, max_duration=60.0
        )
        self.assertEqual(cut_end, 60.0)
        self.assertTrue(hard_cut)

    def test_uses_natural_end_when_video_shorter_than_max_duration(self):
        # Only 45s of video remain after start_time; must not force a cut at
        # 60s that doesn't exist, and must not require a sentence boundary.
        segment_ends = [10.0, 30.0]
        cut_end, hard_cut = clip.choose_cut_end(
            segment_ends, available_duration=45.0, max_duration=60.0
        )
        self.assertEqual(cut_end, 45.0)
        self.assertFalse(hard_cut)

    def test_natural_end_wins_even_with_no_speech_when_video_is_short(self):
        cut_end, hard_cut = clip.choose_cut_end(
            [], available_duration=12.0, max_duration=60.0
        )
        self.assertEqual(cut_end, 12.0)
        self.assertFalse(hard_cut)

    def test_invariant_never_exceeds_max_duration_across_many_cases(self):
        cases = [
            ([1.0, 2.0, 3.0], 90.0),
            ([59.999999], 90.0),
            ([60.0000001], 90.0),
            (list(range(0, 200, 3)), 90.0),
            ([], 200.0),
        ]
        for segment_ends, available_duration in cases:
            cut_end, _ = clip.choose_cut_end(
                segment_ends, available_duration=available_duration, max_duration=60.0
            )
            self.assertLessEqual(cut_end, 60.0)


class TestValidateStartTime(unittest.TestCase):
    def test_accepts_start_time_with_enough_remaining_video(self):
        clip.validate_start_time(0.0, total_duration=120.0)
        clip.validate_start_time(115.0, total_duration=120.0)

    def test_rejects_start_time_too_close_to_end(self):
        with self.assertRaises(clip.ClipValidationError):
            clip.validate_start_time(118.0, total_duration=120.0, min_remaining=3.0)

    def test_rejects_start_time_at_or_past_end(self):
        with self.assertRaises(clip.ClipValidationError):
            clip.validate_start_time(120.0, total_duration=120.0)
        with self.assertRaises(clip.ClipValidationError):
            clip.validate_start_time(130.0, total_duration=120.0)

    def test_rejects_negative_start_time(self):
        with self.assertRaises(clip.ClipValidationError):
            clip.validate_start_time(-1.0, total_duration=120.0)


class TestValidateSourceDuration(unittest.TestCase):
    def test_accepts_duration_within_limit(self):
        clip.validate_source_duration(300.0)

    def test_rejects_duration_over_limit(self):
        with self.assertRaises(clip.ClipValidationError):
            clip.validate_source_duration(clip.MAX_CLIP_SOURCE_DURATION_SECONDS + 1)


class TestSaveClipSourceErrorTranslation(unittest.TestCase):
    """Regression: material_upload's own error types must surface as
    ClipValidationError/ClipServiceError, not bubble up raw and get treated
    as an unexpected server failure by callers (webui/controller only
    special-case clip's own exception types)."""

    def test_undecodable_video_stream_is_a_validation_error_not_a_crash(self):
        with mock.patch.object(
            material_upload,
            "_validate_video",
            side_effect=material_upload.MaterialUploadError(
                "uploaded file must contain a completely decodable video stream"
            ),
        ):
            with self.assertRaises(clip.ClipValidationError) as ctx:
                clip.save_clip_source("episode.mp4", io.BytesIO(b"not a real mp4"))
        self.assertIn("decodable video stream", str(ctx.exception))

    def test_invalid_filename_is_a_validation_error(self):
        with self.assertRaises(clip.ClipValidationError):
            clip.save_clip_source("../evil.mp4", io.BytesIO(b"data"))

    def test_ffmpeg_toolchain_failure_is_a_service_error(self):
        with mock.patch.object(
            material_upload,
            "_validate_video",
            side_effect=material_upload.MaterialServiceError(
                "failed to run FFmpeg for material validation"
            ),
        ):
            with self.assertRaises(clip.ClipServiceError):
                clip.save_clip_source("episode.mp4", io.BytesIO(b"data"))


if __name__ == "__main__":
    unittest.main()
