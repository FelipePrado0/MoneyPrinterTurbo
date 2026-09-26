from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.config import config
from app.services import youtube_metrics, youtube_upload


class FakeHttpError(Exception):
    pass


class _Request:
    def __init__(self, response):
        self._response = response

    def execute(self):
        return self._response


class _Resource:
    def __init__(self, response, calls):
        self._response, self._calls = response, calls

    def list(self, **kwargs):
        self._calls.append(kwargs)
        return _Request(self._response)

    def query(self, **kwargs):
        self._calls.append(kwargs)
        return _Request(self._response)


class FakeData:
    def __init__(self, items):
        self.calls = []
        self._items = items

    def videos(self):
        return _Resource({"items": self._items}, self.calls)


class FakeAnalytics:
    def __init__(self, rows):
        self.calls = []
        self._rows = rows

    def reports(self):
        return _Resource({"rows": self._rows}, self.calls)


def _item(video_id, views, privacy="public", published="2026-09-24T12:00:00Z"):
    return {
        "id": video_id,
        "snippet": {"publishedAt": published},
        "status": {"privacyStatus": privacy},
        "statistics": {"viewCount": str(views), "likeCount": "2", "commentCount": "1"},
    }


@pytest.fixture
def modules(monkeypatch):
    builds = []
    data = FakeData([_item("pub", 120), _item("priv", 0, privacy="private")])

    def build(name, version, **kwargs):
        builds.append(kwargs)
        return data

    fake = SimpleNamespace(build=build, HttpError=FakeHttpError)
    monkeypatch.setattr(youtube_upload, "_load_google_modules", lambda: fake)
    return SimpleNamespace(fake=fake, builds=builds, data=data)


def test_statistics_only_with_api_key_never_touches_oauth(modules, monkeypatch):
    monkeypatch.setitem(config.app, "youtube_data_api_key", "key-1")

    def no_oauth():
        raise AssertionError("OAuth must not be used")

    monkeypatch.setattr(youtube_metrics, "_build_services", no_oauth)
    metrics, problems = youtube_metrics.fetch_metrics(["pub", "priv"], statistics_only=True)

    assert problems == []
    assert modules.builds == [{"developerKey": "key-1", "cache_discovery": False}]
    assert set(modules.data.calls[0]["part"].split(",")) == {"snippet", "statistics", "status"}
    expected_public_at = datetime(2026, 9, 24, 12, tzinfo=timezone.utc).timestamp()
    assert metrics == {
        "pub": {"views": 120, "likes": 2, "comments": 1, "public_at": expected_public_at}
    }


def test_api_key_keeps_statistics_when_oauth_is_unavailable(modules, monkeypatch):
    monkeypatch.setitem(config.app, "youtube_data_api_key", "key-1")

    def broken_oauth():
        raise youtube_metrics.MetricsError("YouTube OAuth is not configured")

    monkeypatch.setattr(youtube_metrics, "_build_services", broken_oauth)
    metrics, problems = youtube_metrics.fetch_metrics(["pub"])

    assert metrics["pub"]["views"] == 120
    assert problems == ["YouTube OAuth is not configured"]


def test_without_api_key_uses_oauth_for_statistics_and_retention(modules, monkeypatch):
    monkeypatch.setitem(config.app, "youtube_data_api_key", "")
    analytics = FakeAnalytics([["pub", 55.5]])
    monkeypatch.setattr(
        youtube_metrics,
        "_build_services",
        lambda: (modules.fake, modules.data, analytics),
    )
    metrics, problems = youtube_metrics.fetch_metrics(["pub"])

    assert problems == []
    assert modules.builds == []
    assert metrics["pub"]["views"] == 120
    assert metrics["pub"]["avg_view_percentage"] == 55.5


def test_statistics_only_skips_retention_query(modules, monkeypatch):
    monkeypatch.setitem(config.app, "youtube_data_api_key", "")
    analytics = FakeAnalytics([["pub", 55.5]])
    monkeypatch.setattr(
        youtube_metrics,
        "_build_services",
        lambda: (modules.fake, modules.data, analytics),
    )
    metrics, _ = youtube_metrics.fetch_metrics(["pub"], statistics_only=True)

    assert analytics.calls == []
    assert "avg_view_percentage" not in metrics["pub"]
