"""Automatic topic selection for the "N times a day, no pre-chosen subject"
schedule mode.

No "will this go viral" oracle exists, and Pexels is a generic stock-footage
library, not a news feed - so this never tries to reproduce a real-world
trending event visually. Instead: pull trending search terms (best-effort,
several independent sources), ask the already-configured LLM to map a term
to a generic visual theme Pexels is likely to have footage for, and fall
back to a fixed rotation of evergreen categories (proven Shorts/Reels
material: nature, animals, curiosities, motivation, food, travel) whenever
trending fails, returns nothing usable, or repeats a recent topic. The
fallback is not optional: trendspy is an unofficial library (its
predecessor, pytrends, was abandoned and stopped working without warning),
so a scheduled dispatch must never depend on it succeeding.
"""

import json
import random

from loguru import logger

from app.config import config
from app.services import llm

# A schedule occurrence with this exact video_subject means "pick a topic
# automatically at dispatch time" instead of using a pre-chosen one. Reuses
# the existing ScheduleOccurrenceInput.video_subject field (required,
# non-empty string) instead of adding a new API/DB field.
AUTO_TOPIC_SENTINEL = "__AUTO_TREND_TOPIC__"

# Proven-performing, visually generic categories with strong Pexels coverage.
# Deliberately avoids anything tied to a specific real person, brand, or
# current event, since stock footage can't depict those.
EVERGREEN_TOPICS = (
    "calm ocean waves at sunset",
    "cute puppies playing in a park",
    "satisfying macro shots of food being cooked",
    "amazing space and galaxy facts",
    "morning motivation and productivity habits",
    "peaceful mountain hiking trails",
    "cats being curious and playful",
    "mesmerizing time-lapse of clouds and weather",
    "simple life hacks for everyday tasks",
    "relaxing rain and forest ambience",
    "colorful street markets around the world",
    "surprising facts about the ocean and marine life",
)

# Below this Jaccard word-overlap ratio, two topics are treated as unrelated.
_SIMILARITY_THRESHOLD = 0.5


def _normalize_topic(topic: str) -> set[str]:
    return set(topic.strip().casefold().split())


def is_similar_to_recent(topic: str, recent_topics: list[str]) -> bool:
    """Word-overlap check so the automatic picker doesn't repeat itself."""
    topic_words = _normalize_topic(topic)
    if not topic_words:
        return False
    for recent in recent_topics:
        recent_words = _normalize_topic(recent)
        if not recent_words:
            continue
        overlap = len(topic_words & recent_words)
        union = len(topic_words | recent_words)
        if union and overlap / union >= _SIMILARITY_THRESHOLD:
            return True
    return False


def pick_evergreen_topic(recent_topics: list[str]) -> str:
    """Random evergreen category, skipping ones too close to recent picks.

    Always returns something: if every category has recently been used
    (small EVERGREEN_TOPICS list, long-running schedule), falls back to a
    plain random pick rather than ever failing a dispatch.
    """
    candidates = [
        topic
        for topic in EVERGREEN_TOPICS
        if not is_similar_to_recent(topic, recent_topics)
    ]
    return random.choice(candidates or list(EVERGREEN_TOPICS))


def fetch_google_trends() -> list[str]:
    """Best-effort trending search terms via trendspy. Never raises."""
    try:
        from trendspy import Trends
    except ImportError:
        logger.debug("trendspy not installed, skipping Google Trends")
        return []
    try:
        trends = Trends()
        results = trends.trending_now()
        return [str(getattr(item, "keyword", item)) for item in results][:20]
    except Exception as exc:
        logger.warning(f"failed to fetch Google Trends: {exc}")
        return []


def fetch_youtube_trending() -> list[str]:
    """Best-effort trending video titles via the official YouTube Data API.

    Optional: only runs if ``app.youtube_data_api_key`` is configured. This
    is a plain read-only API key (public chart data), independent of the
    OAuth credentials used for publishing, so it works even when publishing
    isn't set up.
    """
    api_key = str(config.app.get("youtube_data_api_key", "")).strip()
    if not api_key:
        return []
    try:
        from googleapiclient.discovery import build
    except ImportError:
        logger.debug("google-api-python-client not installed, skipping YouTube trending")
        return []
    try:
        youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        response = youtube.videos().list(
            part="snippet", chart="mostPopular", maxResults=20
        ).execute()
        return [
            item["snippet"]["title"]
            for item in response.get("items", [])
            if item.get("snippet", {}).get("title")
        ]
    except Exception as exc:
        logger.warning(f"failed to fetch YouTube trending: {exc}")
        return []


def map_trend_to_topic(trending_terms: list[str]) -> str | None:
    """Ask the LLM to pick one trending term with strong generic stock-
    footage potential and turn it into a short visual topic. Returns None
    if nothing maps well - the caller falls back to evergreen categories.
    """
    if not trending_terms:
        return None

    prompt = f"""
# Role: Stock-Footage Topic Picker

## Goal
From the trending search terms below, pick AT MOST ONE that could be shown
using generic stock video footage (nature, animals, everyday life, places,
objects, abstract concepts) - NOT tied to a specific real person, brand,
company, sports team, political event, or news story, since none of those
have matching stock footage.

Turn it into a short, concrete, visual topic description (5-10 words)
suitable for a stock-video search, in English.

If NONE of the terms have a plausible generic visual angle, return exactly
the single word: NONE

## Trending terms
{json.dumps(trending_terms, ensure_ascii=False)}

## Output
Return ONLY the topic description, or the word NONE. Nothing else.
""".strip()

    try:
        response = llm._generate_response(prompt).strip()
    except Exception as exc:
        logger.warning(f"failed to map trend to topic via LLM: {exc}")
        return None

    if not response or response.startswith("Error: ") or response.upper() == "NONE":
        return None
    # Guard against a chatty model wrapping the answer in quotes/a sentence.
    return response.strip("\"'.\n ")[:200] or None


def pick_topic(recent_topics: list[str]) -> str:
    """Resolve one automatic topic: real trend if usable, evergreen otherwise.

    Every I/O step (trend fetch, LLM mapping) is best-effort and never
    propagates - a scheduled dispatch must always end up with *some* topic.
    """
    try:
        trending_terms = fetch_google_trends() + fetch_youtube_trending()
    except Exception as exc:
        logger.warning(f"trend fetch failed unexpectedly: {exc}")
        trending_terms = []

    if trending_terms:
        mapped_topic = map_trend_to_topic(trending_terms)
        if mapped_topic and not is_similar_to_recent(mapped_topic, recent_topics):
            return mapped_topic

    return pick_evergreen_topic(recent_topics)
