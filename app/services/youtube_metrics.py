"""Read views/likes/comments and retention for already-published videos.

Statistics are public data: with ``app.youtube_data_api_key`` set they are
read with that plain API key, no OAuth scope needed. Without it they fall
back to the publishing OAuth client, which needs ``youtube.readonly``.
Retention always needs OAuth with ``yt-analytics.readonly``; a token
authorized only for ``youtube.upload`` gets a 403, reported back as
``needs_reauthorization`` so the UI can tell the user exactly what to do.
"""

from datetime import date, datetime

from loguru import logger

from app.config import config
from app.services import youtube_upload

READ_SCOPES = (
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
)
_STATS_BATCH = 50
_ANALYTICS_BATCH = 200
_ANALYTICS_START_DATE = "2020-01-01"


class MetricsError(RuntimeError):
    pass


def _build_services():
    service = youtube_upload.youtube_upload_service
    if not service.is_configured():
        raise MetricsError("YouTube OAuth is not configured")
    modules = youtube_upload._load_google_modules()
    # No ``scopes`` argument: the refresh returns whatever the user granted;
    # asking for more than that fails the refresh itself with invalid_scope.
    credentials = modules.Credentials(
        token=None,
        refresh_token=service.refresh_token,
        client_id=service.client_id,
        client_secret=service.client_secret,
        token_uri=youtube_upload.TOKEN_URI,
    )
    try:
        credentials.refresh(modules.Request())
    except modules.GoogleAuthError as exc:
        raise MetricsError(f"failed to refresh YouTube OAuth credentials: {exc}") from exc
    data = modules.build("youtube", "v3", credentials=credentials, cache_discovery=False)
    analytics = modules.build(
        "youtubeAnalytics", "v2", credentials=credentials, cache_discovery=False
    )
    return modules, data, analytics


def _is_scope_error(exc: Exception) -> bool:
    return youtube_upload._http_error_status(exc) in (401, 403)


def _public_at(item: dict) -> float | None:
    """When the video became public; ``None`` while private or scheduled."""
    if (item.get("status") or {}).get("privacyStatus") != "public":
        return None
    try:
        raw = item["snippet"]["publishedAt"]
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except (KeyError, TypeError, AttributeError, ValueError):
        return None


def fetch_metrics(
    video_ids: list[str], statistics_only: bool = False
) -> tuple[dict[str, dict], list[str]]:
    """Return ``({video_id: metrics}, problems)``; never raises for API errors.

    Only public videos are returned. ``statistics_only`` skips retention and,
    when the API key is set, OAuth entirely. ``problems`` holds
    ``needs_reauthorization`` when a read scope is missing, or short error
    strings for anything else, so one failing source (e.g. Analytics) never
    hides the other (statistics).
    """
    metrics: dict[str, dict] = {vid: {} for vid in video_ids}
    problems: list[str] = []
    if not video_ids:
        return {}, problems
    api_key = str(config.app.get("youtube_data_api_key", "")).strip()
    data = analytics = None
    try:
        modules = youtube_upload._load_google_modules()
        if api_key:
            data = modules.build(
                "youtube", "v3", developerKey=api_key, cache_discovery=False
            )
        if not (api_key and statistics_only):
            _, oauth_data, analytics = _build_services()
            data = data or oauth_data
    except Exception as exc:
        problems.append(str(exc))
        if data is None:
            return {}, problems

    try:
        for start in range(0, len(video_ids), _STATS_BATCH):
            batch = video_ids[start : start + _STATS_BATCH]
            response = (
                data.videos()
                .list(part="snippet,statistics,status", id=",".join(batch))
                .execute()
            )
            for item in response.get("items", []):
                public_at = _public_at(item)
                if public_at is None:
                    continue
                stats = item.get("statistics") or {}
                metrics.setdefault(item["id"], {}).update(
                    views=int(stats.get("viewCount", 0)),
                    likes=int(stats.get("likeCount", 0)),
                    comments=int(stats.get("commentCount", 0)),
                    public_at=public_at,
                )
    except modules.HttpError as exc:
        problems.append(
            "needs_reauthorization" if _is_scope_error(exc) else f"statistics: {exc}"
        )

    if statistics_only or analytics is None:
        if problems:
            logger.warning(f"youtube metrics incomplete: {problems}")
        return {vid: m for vid, m in metrics.items() if m}, problems

    try:
        for start in range(0, len(video_ids), _ANALYTICS_BATCH):
            batch = video_ids[start : start + _ANALYTICS_BATCH]
            response = (
                analytics.reports()
                .query(
                    ids="channel==MINE",
                    startDate=_ANALYTICS_START_DATE,
                    endDate=date.today().isoformat(),
                    metrics="averageViewPercentage",
                    dimensions="video",
                    filters="video==" + ",".join(batch),
                )
                .execute()
            )
            for video_id, avg_view_percentage in response.get("rows") or []:
                metrics.setdefault(video_id, {})["avg_view_percentage"] = float(
                    avg_view_percentage
                )
    except modules.HttpError as exc:
        problem = (
            "needs_reauthorization" if _is_scope_error(exc) else f"analytics: {exc}"
        )
        if problem not in problems:
            problems.append(problem)

    if problems:
        logger.warning(f"youtube metrics incomplete: {problems}")
    return {vid: m for vid, m in metrics.items() if m}, problems
