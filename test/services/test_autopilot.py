import json
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from app.models import const
from app.services import autopilot, schedule_store, video_history


def _topic(subject="Polvo Tem 3 Corações", fact_key="polvo:3-coracoes"):
    return json.dumps(
        {
            "subject": subject,
            "fact_key": fact_key,
            "script": "Gancho. Fato explicado. Comenta aí!",
            "terms": "octopus, ocean, deep sea",
        },
        ensure_ascii=False,
    )


@pytest.fixture
def enabled():
    autopilot.save_settings(
        {"enabled": True, "videos_per_day": 3, "start_time": "09:00", "interval_minutes": 180},
        now=datetime(2026, 9, 25, 6, 0),
    )


def _today_rows(day="2026-09-25"):
    return schedule_store.list_occurrences(group_id=f"autopilot-{day}")


def test_slot_times_spread_from_start_and_drop_past_midnight():
    settings = autopilot.AutopilotSettings(
        videos_per_day=4, start_time="18:00", interval_minutes=240
    )
    slots = autopilot.slot_times(settings, date(2026, 9, 25))
    assert slots == [datetime(2026, 9, 25, 18, 0), datetime(2026, 9, 25, 22, 0)]


def test_plan_day_creates_only_future_slots_once(enabled):
    autopilot.replan_today(now=datetime(2026, 9, 25, 10, 0))
    rows = _today_rows()
    pending = [r for r in rows if r["status"] == schedule_store.STATUS_PENDING]
    assert [r["generate_at"].hour for r in pending] == [12, 15]
    assert pending[0]["video_subject"] == autopilot.AUTO_TOPIC_SENTINEL
    assert autopilot.plan_day(now=datetime(2026, 9, 25, 10, 5)) == 0


def test_disabled_or_paused_plans_nothing():
    now = datetime(2026, 9, 25, 6, 0)
    assert autopilot.plan_day(now=now) == 0
    autopilot.save_settings({"enabled": True}, now=now)
    autopilot.pause(now=now)
    assert all(r["status"] != schedule_store.STATUS_PENDING for r in _today_rows())
    autopilot.resume(now=now)
    assert any(r["status"] == schedule_store.STATUS_PENDING for r in _today_rows())


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        autopilot.save_settings({"videos_per_day": 9})
    with pytest.raises(ValueError):
        autopilot.save_settings({"start_time": "25:00"})


def test_generate_topic_skips_known_fact_key_and_normalizes(enabled):
    video_history.record("old", "Polvo 3 corações", "autopilot", fact_key="polvo:3-coracoes")
    replies = iter(
        [
            _topic(),
            _topic("Tubarões São Mais Velhos que Árvores", "Tubarões: Mais Velhos Que Árvores"),
        ]
    )
    with patch.object(autopilot.llm, "_generate_response", side_effect=lambda p: next(replies)) as gen:
        topic = autopilot.generate_topic(autopilot.load_settings())
    assert topic["fact_key"] == "tubaroes:mais-velhos-que-arvores"
    assert "polvo:3-coracoes" in gen.call_args_list[1].args[0]


def test_generate_topic_gives_up_after_repeated_duplicates(enabled):
    video_history.record("old", "X", "autopilot", fact_key="polvo:3-coracoes")
    with patch.object(autopilot.llm, "_generate_response", return_value=_topic()):
        with pytest.raises(autopilot.AutopilotError):
            autopilot.generate_topic(autopilot.load_settings())


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"approved": true, "reason": ""}', True),
        ('```json\n{"approved": false, "reason": "número errado"}\n```', False),
        ("Error: all OpenRouter models failed", False),
        ("not json at all", False),
    ],
)
def test_fact_check_parsing(reply, expected):
    with patch.object(autopilot.llm, "_generate_response", return_value=reply):
        approved, _ = autopilot.fact_check("S", "script")
    assert approved is expected


def _claim_first(now):
    return schedule_store.claim_due_occurrences(now=now, task_id_factory=lambda: "task-1")[0]


def test_dispatch_success_runs_pipeline_with_generated_content(enabled):
    occurrence = _claim_first(datetime(2026, 9, 25, 9, 1))
    with (
        patch.object(autopilot.llm, "_generate_response", side_effect=[_topic(), '{"approved": true}']),
        patch.object(autopilot.task_service, "start", return_value={"videos": ["v.mp4"]}) as start,
    ):
        autopilot.dispatch(occurrence)
    params = start.call_args.args[1]
    assert params.video_subject == "Polvo Tem 3 Corações"
    assert params.video_script.startswith("Gancho")
    assert params.video_terms == "octopus, ocean, deep sea"
    assert params.video_language == "pt-BR"
    assert callable(start.call_args.kwargs["pre_publish_check"])
    row = video_history.list_videos()[0][0]
    assert row["fact_key"] == "polvo:3-coracoes" and row["source"] == "autopilot"


def test_fact_check_rejection_skips_render_and_schedules_retry(enabled):
    now = datetime(2026, 9, 25, 9, 1)
    occurrence = _claim_first(now)
    with (
        patch.object(autopilot.llm, "_generate_response", side_effect=[_topic(), '{"approved": false, "reason": "falso"}']),
        patch.object(autopilot.task_service, "start") as start,
        patch.object(autopilot, "_now", return_value=now),
    ):
        autopilot.dispatch(occurrence)
    start.assert_not_called()
    assert video_history.list_videos()[0][0]["status"] == video_history.STATUS_REJECTED
    retry = [r for r in _today_rows() if r["params"].get(autopilot.ATTEMPT_KEY) == 2]
    assert len(retry) == 1
    assert retry[0]["generate_at"] == now + timedelta(minutes=10)


def test_last_attempt_failure_alerts_instead_of_retrying(enabled):
    autopilot.save_settings({"max_attempts": 1}, now=datetime(2026, 9, 25, 6, 0))
    occurrence = _claim_first(datetime(2026, 9, 25, 9, 1))
    failed = {"state": const.TASK_STATE_FAILED, "failed_stage": "video", "error": "boom"}
    with (
        patch.object(autopilot.llm, "_generate_response", side_effect=[_topic(), '{"approved": true}']),
        patch.object(autopilot.task_service, "start", return_value=failed),
        patch.object(autopilot.webhook_notifier, "notify_event") as notify,
    ):
        autopilot.dispatch(occurrence)
    notify.assert_called_once()
    assert notify.call_args.args[0] == "autopilot.slot_failed"
    assert video_history.list_videos()[0][0]["status"] == video_history.STATUS_FAILED
    assert not any(r["params"].get(autopilot.ATTEMPT_KEY) for r in _today_rows())


def test_quality_gate_failure_is_recorded_as_rejected(enabled):
    occurrence = _claim_first(datetime(2026, 9, 25, 9, 1))
    failed = {"state": const.TASK_STATE_FAILED, "failed_stage": "quality_gate", "error": "no audio"}
    with (
        patch.object(autopilot.llm, "_generate_response", side_effect=[_topic(), '{"approved": true}']),
        patch.object(autopilot.task_service, "start", return_value=failed),
    ):
        autopilot.dispatch(occurrence)
    assert video_history.list_videos()[0][0]["status"] == video_history.STATUS_REJECTED


def test_quota_reached_skips_generation(enabled):
    occurrence = _claim_first(datetime(2026, 9, 25, 9, 1))
    for i in range(6):
        video_history.record_upload(f"v{i}")
    with patch.object(autopilot.task_service, "start") as start:
        autopilot.dispatch(occurrence)
    start.assert_not_called()
    row = schedule_store.list_occurrences(group_id=occurrence["group_id"])[0]
    assert "quota" in row["error"]


class _FakeClip:
    def __init__(self, duration, audio):
        self.duration, self.audio = duration, audio

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(
    "duration, audio, message",
    [(45, object(), None), (2, object(), "too short"), (45, None, "no audio track"), (400, object(), "too long")],
)
def test_rendered_video_check(tmp_path, duration, audio, message):
    video = tmp_path / "final.mp4"
    video.write_bytes(b"x")
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nOi\n", encoding="utf-8")
    with (
        patch.object(autopilot.video_service, "_open_video_clip_quietly", return_value=_FakeClip(duration, audio)),
        patch.object(autopilot, "_decodes_cleanly", return_value=True),
    ):
        error = autopilot.check_rendered_video([str(video)], str(subtitle), True)
    assert (error is None) if message is None else (message in error)


def test_rendered_video_check_requires_subtitle_when_enabled(tmp_path):
    video = tmp_path / "final.mp4"
    video.write_bytes(b"x")
    with (
        patch.object(autopilot.video_service, "_open_video_clip_quietly", return_value=_FakeClip(45, object())),
        patch.object(autopilot, "_decodes_cleanly", return_value=True),
    ):
        assert "subtitle" in autopilot.check_rendered_video([str(video)], "", True)
        assert autopilot.check_rendered_video([str(video)], "", False) is None


def test_run_metrics_updates_history_and_records_problems():
    video_history.record("t1", "S", "autopilot")
    video_history.mark_published("t1", "yt1", "u")
    with patch.object(
        autopilot.youtube_metrics,
        "fetch_metrics",
        return_value=({"yt1": {"views": 42, "likes": 3, "comments": 1}}, ["needs_reauthorization"]),
    ):
        autopilot.run_metrics()
    assert video_history.list_videos()[0][0]["views"] == 42
    assert video_history.get_state("metrics_problems") == ["needs_reauthorization"]


def test_tick_runs_metrics_and_backup_once_per_day():
    submitted = []
    now = datetime(2026, 9, 25, 6, 0)
    autopilot.tick(now, submitted.append)
    autopilot.tick(now + timedelta(minutes=1), submitted.append)
    assert submitted == [autopilot.run_metrics, autopilot.run_snapshots, autopilot.run_backup]


def test_tick_runs_snapshots_once_per_hour():
    submitted = []
    now = datetime(2026, 9, 25, 6, 0)
    for minutes in (0, 30, 60, 61):
        autopilot.tick(now + timedelta(minutes=minutes), submitted.append)
    assert submitted.count(autopilot.run_snapshots) == 2


def test_run_snapshots_fetches_statistics_of_candidates_only():
    now = datetime.now().timestamp()
    video_history.record("t1", "S", "autopilot")
    video_history.mark_published("t1", "yt1", "u")
    calls = []

    def fake_fetch(video_ids, statistics_only=False):
        calls.append((video_ids, statistics_only))
        return {"yt1": {"views": 70, "public_at": now - 86400 - 60}}, []

    with patch.object(autopilot.youtube_metrics, "fetch_metrics", side_effect=fake_fetch):
        autopilot.run_snapshots()
    assert calls == [(["yt1"], True)]
    assert video_history.list_videos()[0][0]["views_24h"] == 70


def test_run_snapshots_records_problems_without_writing_zeros():
    video_history.record("t1", "S", "autopilot")
    video_history.mark_published("t1", "yt1", "u")
    with patch.object(
        autopilot.youtube_metrics, "fetch_metrics", return_value=({}, ["statistics: boom"])
    ):
        autopilot.run_snapshots()
    row = video_history.list_videos()[0][0]
    assert row["views_24h"] is None and row["views"] is None
    assert video_history.get_state("metrics_problems") == ["statistics: boom"]


def test_topic_prompt_lists_24h_performers():
    now = datetime.now().timestamp()
    for index, views in enumerate([900, 10]):
        video_history.record(f"t{index}", f"Tema {index}", "autopilot")
        video_history.mark_published(f"t{index}", f"y{index}", "u")
        video_history.update_metrics(f"y{index}", views=views, public_at=now - 86400 - 60)
    prompt = autopilot._topic_prompt(autopilot.AutopilotSettings(), [])
    assert "- Tema 0 (900 views in the first 24h)" in prompt
    assert "- Tema 1 (10 views in the first 24h)" in prompt


def test_first_tick_backfills_dispatched_schedule_topics_once():
    schedule_store.create_schedule(
        [{"generate_at": datetime(2026, 9, 1, 9, 0), "video_subject": "Polvo Tem 3 Corações"}],
        {"video_subject": "x"},
    )
    schedule_store.claim_due_occurrences(now=datetime(2026, 9, 2), task_id_factory=lambda: "old-task")
    autopilot.tick(datetime(2026, 9, 25, 1, 0), lambda fn: None)
    autopilot.tick(datetime(2026, 9, 25, 1, 1), lambda fn: None)
    assert video_history.known_topics() == [("Polvo Tem 3 Corações", None)]
