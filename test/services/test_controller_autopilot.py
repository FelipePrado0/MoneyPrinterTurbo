from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import asgi
from app.config import config
from app.services import video_history


@pytest.fixture
def client():
    saved = dict(config.app)
    config.app["api_key"] = ""
    yield TestClient(asgi.app)
    config.app.clear()
    config.app.update(saved)


def test_get_and_update_config(client):
    response = client.get("/api/v1/autopilot/config")
    assert response.status_code == 200
    assert response.json()["data"]["videos_per_day"] == 3

    response = client.put(
        "/api/v1/autopilot/config", json={"videos_per_day": 5, "niche": "espaço"}
    )
    assert response.status_code == 200
    data = client.get("/api/v1/autopilot/config").json()["data"]
    assert data["videos_per_day"] == 5 and data["niche"] == "espaço"


@pytest.mark.parametrize(
    "body", [{"videos_per_day": 7}, {"start_time": "9h"}, {"unknown": 1}]
)
def test_update_config_rejects_invalid_values(client, body):
    response = client.put("/api/v1/autopilot/config", json=body)
    assert response.status_code in (400, 422)
    assert client.get("/api/v1/autopilot/config").json()["data"]["videos_per_day"] == 3


def test_pause_resume_and_status(client):
    assert client.post("/api/v1/autopilot/pause").status_code == 200
    assert client.get("/api/v1/autopilot/status").json()["data"]["paused"] is True
    assert client.post("/api/v1/autopilot/resume").status_code == 200
    status = client.get("/api/v1/autopilot/status").json()["data"]
    assert status["paused"] is False
    assert {"uploads_today", "today", "next_run", "settings"} <= set(status)


def test_video_history_is_paginated(client):
    for i in range(3):
        video_history.record(f"t{i}", f"S{i}", "autopilot")
    response = client.get("/api/v1/videos", params={"limit": 2, "offset": 0})
    body = response.json()["data"]
    assert response.status_code == 200
    assert body["total"] == 3 and len(body["items"]) == 2
    assert client.get("/api/v1/videos", params={"limit": 999}).status_code in (400, 422)
    assert client.get("/api/v1/videos", params={"status": "bogus"}).status_code in (400, 422)


def test_free_models_endpoint(client):
    rows = [{"id": "m:free", "context_length": 1, "state": "ok", "cooldown_until": None}]
    with patch("app.controllers.v1.autopilot.llm_free_models.status", return_value=rows):
        response = client.get("/api/v1/llm/free-models")
    assert response.status_code == 200
    assert response.json()["data"] == rows


def test_endpoints_require_api_key_when_configured(client):
    config.app["api_key"] = "secret"
    assert client.get("/api/v1/autopilot/status").status_code == 401
    ok = client.get("/api/v1/autopilot/status", headers={"x-api-key": "secret"})
    assert ok.status_code == 200
