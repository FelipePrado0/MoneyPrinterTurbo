"""Tests for app.services.trend_topic: automatic topic selection for the
"N times a day, no pre-chosen subject" schedule mode.

Pure decision logic (similarity check, evergreen fallback selection,
orchestration branching) is tested here without touching the network, an
LLM, or trendspy/YouTube - those are mocked. Real trend-fetching/LLM-mapping
behavior is out of scope for automated tests (matches how the clip feature's
STT/ffmpeg-touching code is only exercised through mocks in this suite).
"""

import unittest
from unittest import mock

from app.services import trend_topic


class TestIsSimilarToRecent(unittest.TestCase):
    def test_exact_match_is_similar(self):
        self.assertTrue(
            trend_topic.is_similar_to_recent(
                "ocean waves at sunset", ["ocean waves at sunset"]
            )
        )

    def test_case_and_whitespace_insensitive(self):
        self.assertTrue(
            trend_topic.is_similar_to_recent(
                "  Ocean Waves At Sunset  ", ["ocean waves at sunset"]
            )
        )

    def test_high_word_overlap_is_similar(self):
        self.assertTrue(
            trend_topic.is_similar_to_recent(
                "calm ocean waves at sunset beach",
                ["ocean waves at sunset"],
            )
        )

    def test_unrelated_topic_is_not_similar(self):
        self.assertFalse(
            trend_topic.is_similar_to_recent(
                "astronaut training in space",
                ["ocean waves at sunset", "cute puppies playing"],
            )
        )

    def test_empty_recent_list_is_never_similar(self):
        self.assertFalse(trend_topic.is_similar_to_recent("anything", []))


class TestPickEvergreenTopic(unittest.TestCase):
    def test_always_returns_a_non_empty_topic(self):
        topic = trend_topic.pick_evergreen_topic([])
        self.assertIsInstance(topic, str)
        self.assertTrue(topic.strip())

    def test_avoids_recently_used_topics_when_alternatives_exist(self):
        recent = list(trend_topic.EVERGREEN_TOPICS[:-1])
        topic = trend_topic.pick_evergreen_topic(recent)
        self.assertEqual(topic, trend_topic.EVERGREEN_TOPICS[-1])

    def test_never_raises_even_when_everything_was_recently_used(self):
        # All categories exhausted - must still return something, not crash
        # a scheduled dispatch.
        topic = trend_topic.pick_evergreen_topic(list(trend_topic.EVERGREEN_TOPICS))
        self.assertIsInstance(topic, str)
        self.assertTrue(topic.strip())


class TestPickTopic(unittest.TestCase):
    """Orchestration branching: trending -> LLM mapping -> evergreen fallback."""

    def test_uses_mapped_trend_when_available_and_not_repeated(self):
        with mock.patch.object(
            trend_topic, "fetch_google_trends", return_value=["heat wave"]
        ), mock.patch.object(
            trend_topic, "fetch_youtube_trending", return_value=[]
        ), mock.patch.object(
            trend_topic,
            "map_trend_to_topic",
            return_value="sun-drenched desert dunes",
        ) as mocked_map, mock.patch.object(
            trend_topic, "pick_evergreen_topic"
        ) as mocked_evergreen:
            topic = trend_topic.pick_topic(recent_topics=[])

        self.assertEqual(topic, "sun-drenched desert dunes")
        mocked_map.assert_called_once()
        mocked_evergreen.assert_not_called()

    def test_falls_back_to_evergreen_when_no_trends_found(self):
        with mock.patch.object(
            trend_topic, "fetch_google_trends", return_value=[]
        ), mock.patch.object(
            trend_topic, "fetch_youtube_trending", return_value=[]
        ), mock.patch.object(
            trend_topic, "map_trend_to_topic"
        ) as mocked_map, mock.patch.object(
            trend_topic, "pick_evergreen_topic", return_value="calm ocean waves"
        ) as mocked_evergreen:
            topic = trend_topic.pick_topic(recent_topics=[])

        self.assertEqual(topic, "calm ocean waves")
        mocked_map.assert_not_called()
        mocked_evergreen.assert_called_once()

    def test_falls_back_to_evergreen_when_llm_mapping_fails(self):
        with mock.patch.object(
            trend_topic, "fetch_google_trends", return_value=["some trend"]
        ), mock.patch.object(
            trend_topic, "fetch_youtube_trending", return_value=[]
        ), mock.patch.object(
            trend_topic, "map_trend_to_topic", return_value=None
        ), mock.patch.object(
            trend_topic, "pick_evergreen_topic", return_value="cute puppies playing"
        ) as mocked_evergreen:
            topic = trend_topic.pick_topic(recent_topics=[])

        self.assertEqual(topic, "cute puppies playing")
        mocked_evergreen.assert_called_once()

    def test_falls_back_to_evergreen_when_mapped_topic_repeats_recent(self):
        with mock.patch.object(
            trend_topic, "fetch_google_trends", return_value=["some trend"]
        ), mock.patch.object(
            trend_topic, "fetch_youtube_trending", return_value=[]
        ), mock.patch.object(
            trend_topic, "map_trend_to_topic", return_value="ocean waves at sunset"
        ), mock.patch.object(
            trend_topic, "pick_evergreen_topic", return_value="cute puppies playing"
        ) as mocked_evergreen:
            topic = trend_topic.pick_topic(
                recent_topics=["ocean waves at sunset"]
            )

        self.assertEqual(topic, "cute puppies playing")
        mocked_evergreen.assert_called_once()

    def test_trend_source_exceptions_never_propagate(self):
        # A trend source breaking (as pytrends did) must degrade to the
        # evergreen fallback, never crash the scheduled dispatch.
        with mock.patch.object(
            trend_topic, "fetch_google_trends", side_effect=RuntimeError("boom")
        ), mock.patch.object(
            trend_topic, "fetch_youtube_trending", return_value=[]
        ), mock.patch.object(
            trend_topic, "pick_evergreen_topic", return_value="cute puppies playing"
        ):
            topic = trend_topic.pick_topic(recent_topics=[])

        self.assertEqual(topic, "cute puppies playing")


if __name__ == "__main__":
    unittest.main()
