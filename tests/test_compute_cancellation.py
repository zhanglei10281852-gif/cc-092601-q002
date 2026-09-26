from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.compute.service import ComputeOperationsService
from app.core.clock import FrozenClock, to_storage
from app.core.errors import ConflictError
from app.database import close_connection, get_connection, init_db

TEMPLATE = {
    "code": "solver-a",
    "name": "方程求解模板",
    "algorithm": "solver-a",
    "parameter_schema": {
        "iterations": {"type": "integer", "required": True, "minimum": 1, "maximum": 10000},
        "mode": {"type": "string", "required": True, "choices": ["fast", "accurate"]},
    },
    "default_parameters": {},
    "max_runtime_seconds": 300,
    "max_attempts": 2,
}


def submit_payload(key: str, *, user: str = "researcher-1") -> dict:
    return {
        "template_code": "solver-a",
        "project_code": "project-a",
        "requested_by": user,
        "parameters": {"iterations": 100, "mode": "accurate"},
        "priority": 50,
        "idempotency_key": key,
    }


@pytest.fixture()
def service(tmp_path: Path):
    os.environ["TOWNSHIP_DATABASE_PATH"] = str(tmp_path / "cancel.db")
    close_connection()
    init_db()
    clock = FrozenClock(datetime(2026, 9, 26, 8, 0, tzinfo=UTC))
    instance = ComputeOperationsService(get_connection(), clock)
    instance.create_template(TEMPLATE, "administrator")
    yield instance, clock
    close_connection()


def sqlite_states(user: str = "researcher-1") -> dict[str, int]:
    rows = get_connection().execute("SELECT status,COUNT(*) AS amount FROM compute_tasks WHERE requested_by=? GROUP BY status", (user,)).fetchall()
    return {str(row["status"]): int(row["amount"]) for row in rows}


def test_running_cancel_converges_on_timely_worker_confirmation(service):
    instance, clock = service
    instance.set_quota({"subject_type": "user", "subject_key": "researcher-1", "max_queued": 10, "max_running": 1, "daily_submissions": 10}, "administrator")
    task = instance.submit(submit_payload("cancel-timely-01"))
    claimed = instance.claim("worker-a", ["solver-a"], 60)
    assert claimed and claimed["id"] == task["id"]

    requested = instance.cancel(task["id"], "administrator", "课题负责人撤回输入错误的算例")
    assert requested["status"] == "cancel_requested"
    assert requested["cancel_requested_by"] == "administrator"
    assert requested["cancel_reason"] == "课题负责人撤回输入错误的算例"
    assert requested["cancel_requested_at"] == to_storage(clock.now())
    assert requested["finished_at"] is None
    assert requested["lease_owner"] == "worker-a"

    # 取消请求未收敛前仍占用运行配额，接口拒绝与 SQLite 占用一致。
    with pytest.raises(ConflictError):
        instance.submit(submit_payload("cancel-timely-02"))
    assert sqlite_states().get("cancel_requested") == 1

    # 重复取消返回同一业务结果：状态、版本、取消时间不变，不新增干预记录。
    repeated = instance.cancel(task["id"], "administrator", "课题负责人撤回输入错误的算例")
    assert repeated == requested
    interventions = instance.get_task(task["id"])["interventions"]
    assert [item["action"] for item in interventions] == ["cancel"]

    # 其他工作者不能代为确认。
    with pytest.raises(ConflictError):
        instance.confirm_cancel(task["id"], "worker-b")

    clock.advance(seconds=5)
    confirmed = instance.confirm_cancel(task["id"], "worker-a")
    assert confirmed["status"] == "cancelled"
    assert confirmed["cancel_confirmation_source"] == "worker"
    assert confirmed["cancel_confirmed_by"] == "worker-a"
    assert confirmed["cancelled_at"] == to_storage(clock.now())
    assert confirmed["finished_at"] == confirmed["cancelled_at"]
    assert confirmed["lease_owner"] == ""
    assert [item["action"] for item in instance.get_task(task["id"])["interventions"]] == ["cancel", "cancel_confirm"]

    # 确认后工作者停止上报普通结果。
    with pytest.raises(ConflictError):
        instance.complete(task["id"], "worker-a", {"value": 1}, {})
    assert instance.get_task(task["id"])["results"] == []

    # 终态释放配额：接口接受新提交，SQLite 中占用同步消失。
    follow_up = instance.submit(submit_payload("cancel-timely-02"))
    assert follow_up["status"] == "queued"
    states = sqlite_states()
    assert states.get("cancel_requested", 0) == 0
    assert states["cancelled"] == 1 and states["queued"] == 1


def test_running_cancel_settles_via_recovery_when_worker_is_lost(service):
    instance, clock = service
    task = instance.submit(submit_payload("cancel-lost-0001"))
    instance.claim("worker-b", ["solver-a"], 30)
    requested = instance.cancel(task["id"], "administrator", "管理员运行中取消")
    assert requested["status"] == "cancel_requested"

    clock.advance(seconds=31)
    outcome = instance.recover_expired()
    assert outcome["cancelled"] == [task["id"]]
    assert outcome["recovered"] == [] and outcome["exhausted"] == []

    details = instance.get_task(task["id"])
    assert details["status"] == "cancelled"
    assert details["cancel_requested_by"] == "administrator"
    assert details["cancel_reason"] == "管理员运行中取消"
    assert details["cancel_confirmation_source"] == "recovery"
    assert details["cancel_confirmed_by"] == "recovery-worker"
    assert details["cancelled_at"] == to_storage(clock.now())
    assert details["finished_at"] == details["cancelled_at"]
    assert [item["action"] for item in details["interventions"]] == ["cancel", "cancel_recovery"]

    # 失联工作者迟到的心跳、确认与普通结果都不能改变终态。
    with pytest.raises(ConflictError):
        instance.heartbeat(task["id"], "worker-b", 30)
    late = instance.confirm_cancel(task["id"], "worker-b")
    assert late["status"] == "cancelled"
    assert late["cancel_confirmation_source"] == "recovery"
    with pytest.raises(ConflictError):
        instance.complete(task["id"], "worker-b", {"value": 1}, {})

    # 队列统计与 SQLite 占用不再把该任务算作未结束。
    summary = instance.summary()
    assert summary["states"].get("cancel_requested", 0) == 0
    assert summary["states"]["cancelled"] == 1
    assert sqlite_states() == {"cancelled": 1}


def test_cancel_and_complete_race_yields_single_terminal_state(service):
    instance, clock = service
    # 成功回执先到：取消只能以冲突告终，任务保持 succeeded。
    first = instance.submit(submit_payload("cancel-race-0001"))
    instance.claim("worker-c", ["solver-a"], 60)
    done = instance.complete(first["id"], "worker-c", {"value": 42}, {"seconds": 3})
    assert done["status"] == "succeeded"
    with pytest.raises(ConflictError):
        instance.cancel(first["id"], "administrator", "来晚的取消")
    assert instance.get_task(first["id"])["status"] == "succeeded"

    # 取消先到：普通结果回执被拒绝，确认后只能落为 cancelled。
    second = instance.submit(submit_payload("cancel-race-0002"))
    instance.claim("worker-c", ["solver-a"], 60)
    instance.cancel(second["id"], "administrator", "先到的取消")
    with pytest.raises(ConflictError):
        instance.complete(second["id"], "worker-c", {"value": 43}, {})
    settled = instance.confirm_cancel(second["id"], "worker-c")
    assert settled["status"] == "cancelled"
    details = instance.get_task(second["id"])
    assert details["results"] == []
    assert details["current_result_version"] is None
    assert sqlite_states() == {"succeeded": 1, "cancelled": 1}


def test_queued_cancel_settles_immediately_and_terminal_guards(service):
    instance, clock = service
    task = instance.submit(submit_payload("cancel-queued-01"))
    cancelled = instance.cancel(task["id"], "project-lead", "撤回输入错误")
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_requested_by"] == "project-lead"
    assert cancelled["cancel_reason"] == "撤回输入错误"
    assert cancelled["cancel_confirmation_source"] == "immediate"
    assert cancelled["cancel_confirmed_by"] == "project-lead"
    assert cancelled["cancelled_at"] == cancelled["finished_at"] == to_storage(clock.now())

    repeated = instance.cancel(task["id"], "project-lead", "撤回输入错误")
    assert repeated == cancelled
    interventions = instance.get_task(task["id"])["interventions"]
    assert [item["action"] for item in interventions] == ["cancel"]

    # 已取消任务可以人工重试，重试后取消元数据被清空。
    retried = instance.retry(task["id"], "administrator", "参数修正后重跑")
    assert retried["status"] == "queued"
    assert retried["cancel_requested_by"] == "" and retried["cancelled_at"] == ""

    # 失败任务进入终态后不允许取消。
    instance.claim("worker-d", ["solver-a"], 30)
    failed = instance.fail(task["id"], "worker-d", "numeric_error", "数值不收敛", False)
    assert failed["status"] == "failed"
    with pytest.raises(ConflictError):
        instance.cancel(task["id"], "administrator", "终态不可取消")


def test_cancel_protocol_over_http_matches_sqlite(client):
    from app.database import get_connection as connection

    created = client.post("/api/compute/templates?actor=administrator", json=TEMPLATE)
    assert created.status_code == 201, created.text
    quota = client.put(
        "/api/compute/quotas?actor=administrator",
        json={"subject_type": "user", "subject_key": "researcher-1", "max_queued": 10, "max_running": 1, "daily_submissions": 10},
    )
    assert quota.status_code == 200
    submitted = client.post("/api/compute/tasks", json=submit_payload("cancel-http-0001"))
    assert submitted.status_code == 202
    task_id = submitted.json()["id"]
    claimed = client.post("/api/compute/tasks/claim", json={"worker_id": "worker-http", "capabilities": ["solver-a"], "lease_seconds": 60})
    assert claimed.json()["task"]["id"] == task_id

    cancelled = client.post(f"/api/compute/tasks/{task_id}/cancel", json={"actor": "administrator", "reason": "课题负责人撤回算例"})
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancel_requested"

    # 取消请求未收敛：运行配额被占用，重复提交被接口拒绝。
    blocked = client.post("/api/compute/tasks", json=submit_payload("cancel-http-0002"))
    assert blocked.status_code == 409

    # 重复取消返回同一业务结果。
    repeated = client.post(f"/api/compute/tasks/{task_id}/cancel", json={"actor": "administrator", "reason": "课题负责人撤回算例"})
    assert repeated.status_code == 200
    assert repeated.json()["version"] == cancelled.json()["version"]
    assert repeated.json()["cancel_requested_at"] == cancelled.json()["cancel_requested_at"]

    # 工作者确认取消后，普通结果回执被拒绝。
    confirmed = client.post(f"/api/compute/tasks/{task_id}/cancel-confirm", json={"worker_id": "worker-http"})
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "cancelled"
    rejected = client.post(f"/api/compute/tasks/{task_id}/complete", json={"worker_id": "worker-http", "result": {"value": 1}, "metrics": {}})
    assert rejected.status_code == 409

    # 终态、发起人、原因、确认来源和时间均可查询。
    detail = client.get(f"/api/compute/task-details/{task_id}").json()
    assert detail["status"] == "cancelled"
    assert detail["cancel_requested_by"] == "administrator"
    assert detail["cancel_reason"] == "课题负责人撤回算例"
    assert detail["cancel_confirmation_source"] == "worker"
    assert detail["cancel_confirmed_by"] == "worker-http"
    assert detail["cancel_requested_at"] and detail["cancelled_at"]
    assert [item["action"] for item in detail["interventions"]] == ["cancel", "cancel_confirm"]

    # 接口返回与 SQLite 中的配额占用一致：终态释放运行配额。
    states = {row["status"]: row["amount"] for row in connection().execute("SELECT status,COUNT(*) AS amount FROM compute_tasks GROUP BY status").fetchall()}
    assert states.get("cancel_requested", 0) == 0
    assert states["cancelled"] == 1
    accepted = client.post("/api/compute/tasks", json=submit_payload("cancel-http-0002"))
    assert accepted.status_code == 202
    summary = client.get("/api/compute/summary").json()
    assert summary["states"].get("cancel_requested", 0) == 0
    assert summary["states"]["cancelled"] == 1
