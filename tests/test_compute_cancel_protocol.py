from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.compute.service import ComputeOperationsService
from app.core.clock import FrozenClock
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

START = datetime(2026, 9, 26, 2, 0, tzinfo=UTC)


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
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWNSHIP_DATABASE_PATH", str(tmp_path / "cancel.db"))
    close_connection()
    init_db()
    clock = FrozenClock(START)
    instance = ComputeOperationsService(get_connection(), clock)
    instance.create_template(TEMPLATE, "administrator")
    yield instance
    close_connection()


def claim_and_run(instance: ComputeOperationsService, key: str, *, user: str = "researcher-1", lease: int = 60) -> dict:
    task = instance.submit(submit_payload(key, user=user))
    claimed = instance.claim("worker-1", ["solver-a"], lease)
    assert claimed is not None and claimed["id"] == task["id"]
    return claimed


# ---------- 场景一：工作者及时确认 ----------

def test_worker_confirms_cancel_in_time(service):
    task = claim_and_run(service, "cancel-timely-001")

    requested = service.cancel(task["id"], "pi-lead", "输入参数错误，撤回算例")
    assert requested["status"] == "cancel_requested"
    assert requested["finished_at"] is None

    # 心跳会告知工作者取消已挂起，工作者应停止普通上报。
    with pytest.raises(ConflictError):
        service.heartbeat(task["id"], "worker-1", 60)

    confirmed = service.confirm_cancel(task["id"], "worker-1")
    assert confirmed["status"] == "cancelled"
    assert confirmed["finished_at"] == confirmed["updated_at"]

    record = service.get_cancel(task["id"])
    assert record["terminal"] is True
    assert record["cancel_requested_by"] == "pi-lead"
    assert record["cancel_reason"] == "输入参数错误，撤回算例"
    assert record["cancelled_by"] == "pi-lead"          # 终态归属发起人
    assert record["cancel_confirmed_by"] == "worker-1"  # 实际确认者
    assert record["cancel_confirm_source"] == "worker"
    assert record["cancel_requested_at"] == "2026-09-26T02:00:00+00:00"
    assert record["cancelled_at"] == "2026-09-26T02:00:00+00:00"

    # 终态之后不再接收普通结果
    with pytest.raises(ConflictError):
        service.complete(task["id"], "worker-1", {"value": 1.0}, {})
    assert service.repository.result_versions(task["id"]) == []


def test_repeated_cancel_and_confirm_return_same_business_result(service):
    task = claim_and_run(service, "cancel-repeat-001")
    first = service.cancel(task["id"], "pi-lead", "首次取消原因")
    second = service.cancel(task["id"], "someone-else", "另一个取消原因")
    # 同一业务结果：同一状态、同一版本、首次发起人与原因保留
    assert second["status"] == first["status"] == "cancel_requested"
    assert second["version"] == first["version"]
    assert second["cancel_requested_by"] == "pi-lead"
    assert second["cancel_reason"] == "首次取消原因"

    confirmed = service.confirm_cancel(task["id"], "worker-1")
    confirmed_again = service.confirm_cancel(task["id"], "worker-1")
    assert confirmed_again["status"] == confirmed["status"] == "cancelled"
    assert confirmed_again["version"] == confirmed["version"]
    third_cancel = service.cancel(task["id"], "pi-lead", "首次取消原因")
    assert third_cancel["status"] == "cancelled"
    assert third_cancel["version"] == confirmed["version"]

    details = service.get_task(task["id"])
    assert [item["action"] for item in details["interventions"]] == ["cancel", "cancel_confirm"]


def test_queued_cancel_is_terminal_directly(service):
    task = service.submit(submit_payload("cancel-queued-001"))
    cancelled = service.cancel(task["id"], "pi-lead", "项目暂停")
    assert cancelled["status"] == "cancelled"
    record = service.get_cancel(task["id"])
    assert record["cancel_confirm_source"] == "direct"
    assert record["cancelled_by"] == record["cancel_requested_by"] == "pi-lead"
    assert service.cancel(task["id"], "pi-lead", "项目暂停")["version"] == cancelled["version"]


# ---------- 场景二：工作者失联，租约恢复收敛 ----------

def test_lost_worker_cancel_converges_after_lease_expiry(service):
    task = claim_and_run(service, "cancel-lost-001", lease=10)
    service.cancel(task["id"], "pi-lead", "撤回错误算例")

    # 租约未过期：恢复入口不处理挂起的取消
    service.clock.advance(seconds=9)
    assert service.recover_expired()["cancelled"] == []
    assert service.get_task(task["id"])["status"] == "cancel_requested"

    # 超过租约：自动落为已取消
    service.clock.advance(seconds=2)
    outcome = service.recover_expired()
    assert outcome["cancelled"] == [task["id"]]

    record = service.get_cancel(task["id"])
    assert record["status"] == "cancelled"
    assert record["terminal"] is True
    assert record["cancel_confirm_source"] == "recovery"
    assert record["cancel_confirmed_by"] == "recovery-worker"
    assert record["cancelled_by"] == "pi-lead"
    assert record["cancelled_at"] == "2026-09-26T02:00:11+00:00"

    details = service.get_task(task["id"])
    assert details["interventions"][-1]["action"] == "lease_recovery"

    # 再次恢复幂等：不会重复处理
    service.clock.advance(seconds=60)
    assert service.recover_expired()["cancelled"] == []


# ---------- 场景三：取消与成功回执竞争，只能有一个终态 ----------

def test_cancel_before_complete_rejects_result_and_confirms_cancel(service):
    task = claim_and_run(service, "cancel-race-a-001")
    service.cancel(task["id"], "pi-lead", "撤回")
    with pytest.raises(ConflictError) as exc_info:
        service.complete(task["id"], "worker-1", {"value": 9.9}, {"seconds": 3})
    assert exc_info.value.context["task"]["status"] == "cancelled"

    record = service.get_cancel(task["id"])
    assert record["status"] == "cancelled"
    assert record["cancel_confirm_source"] == "complete"
    assert record["cancel_confirmed_by"] == "worker-1"
    assert service.repository.result_versions(task["id"]) == []


def test_complete_before_cancel_wins_single_terminal_state(service):
    task = claim_and_run(service, "cancel-race-b-001")
    completed = service.complete(task["id"], "worker-1", {"value": 1.0}, {"seconds": 1})
    assert completed["status"] == "succeeded"

    with pytest.raises(ConflictError) as exc_info:
        service.cancel(task["id"], "pi-lead", "撤回太晚")
    assert exc_info.value.context["task"]["status"] == "succeeded"

    record = service.get_cancel(task["id"])
    assert record["status"] == "succeeded"
    assert record["terminal"] is True
    assert record["cancel_requested_at"] is None
    details = service.get_task(task["id"])
    assert len(details["results"]) == 1
    assert [item["action"] for item in details["interventions"]] == []


def test_simultaneous_arrival_same_timestamp_single_terminal(service):
    """同一时刻到达：先提交者赢，另一个返回冲突，接口状态与库内终态一致。"""
    task = claim_and_run(service, "cancel-race-c-001")
    service.cancel(task["id"], "pi-lead", "撤回")  # 与完成同一秒
    with pytest.raises(ConflictError):
        service.complete(task["id"], "worker-1", {"value": 1.0}, {})
    row = service.connection.execute("SELECT status,finished_at,cancel_confirm_source FROM compute_tasks WHERE id=?", (task["id"],)).fetchone()
    assert row["status"] == "cancelled"
    assert row["finished_at"] == "2026-09-26T02:00:00+00:00"
    assert row["cancel_confirm_source"] == "complete"


# ---------- 接口返回与 SQLite 配额占用一致 ----------

def test_quota_release_matches_database_after_worker_confirm(service):
    service.set_quota(
        {"subject_type": "user", "subject_key": "limited", "max_queued": 5, "max_running": 1, "daily_submissions": 100},
        "administrator",
    )
    task = claim_and_run(service, "quota-cancel-001", user="limited")
    # running 已占满，同用户新任务在提交环节即被配额拒绝
    with pytest.raises(ConflictError):
        service.submit(submit_payload("quota-cancel-002", user="limited"))

    service.cancel(task["id"], "pi-lead", "撤回")
    service.confirm_cancel(task["id"], "worker-1")

    # 接口视图
    states = service.summary()["states"]
    # SQLite 实际占用
    db_states = {
        row["status"]: row["amount"]
        for row in service.connection.execute("SELECT status,COUNT(*) AS amount FROM compute_tasks GROUP BY status").fetchall()
    }
    assert states == db_states
    assert states.get("running", 0) == 0
    assert states.get("cancel_requested", 0) == 0
    assert states["cancelled"] >= 1

    # 取消不再占用 running 配额：新任务可以提交并被领取
    second = service.submit(submit_payload("quota-cancel-002", user="limited"))
    next_task = service.claim("worker-2", ["solver-a"], 60)
    assert next_task is not None and next_task["id"] == second["id"]


def test_quota_release_matches_database_after_lease_recovery(service):
    service.set_quota(
        {"subject_type": "user", "subject_key": "limited", "max_queued": 5, "max_running": 1, "daily_submissions": 100},
        "administrator",
    )
    task = claim_and_run(service, "quota-lost-001", user="limited", lease=10)
    with pytest.raises(ConflictError):
        service.submit(submit_payload("quota-lost-002", user="limited"))

    service.cancel(task["id"], "pi-lead", "撤回")
    service.clock.advance(seconds=11)
    service.recover_expired()

    states = service.summary()["states"]
    db_states = {
        row["status"]: row["amount"]
        for row in service.connection.execute("SELECT status,COUNT(*) AS amount FROM compute_tasks GROUP BY status").fetchall()
    }
    assert states == db_states
    assert states.get("running", 0) == 0
    second = service.submit(submit_payload("quota-lost-002", user="limited"))
    assert service.claim("worker-2", ["solver-a"], 60)["id"] == second["id"]


# ---------- HTTP 接口 ----------

def test_cancel_protocol_http_endpoints(client):
    response = client.post("/api/compute/templates?actor=administrator", json=TEMPLATE)
    assert response.status_code == 201, response.text
    task = client.post("/api/compute/tasks", json=submit_payload("http-cancel-001")).json()
    client.post("/api/compute/tasks/claim", json={"worker_id": "w1", "capabilities": ["solver-a"], "lease_seconds": 60})

    cancelled_request = client.post(f"/api/compute/tasks/{task['id']}/cancel", json={"actor": "pi-lead", "reason": "撤回错误算例"})
    assert cancelled_request.status_code == 200
    assert cancelled_request.json()["status"] == "cancel_requested"

    late_result = client.post(
        f"/api/compute/tasks/{task['id']}/complete",
        json={"worker_id": "w1", "result": {"value": 1}, "metrics": {}},
    )
    assert late_result.status_code == 409
    assert late_result.json()["error"]["context"]["task"]["status"] == "cancelled"

    # 该任务已被回执顺带确认，重复显式确认返回同一终态
    again = client.post(f"/api/compute/tasks/{task['id']}/cancel/confirm", json={"worker_id": "w1"})
    assert again.status_code == 200 and again.json()["status"] == "cancelled"

    view = client.get(f"/api/compute/tasks/{task['id']}/cancel")
    assert view.status_code == 200
    body = view.json()
    assert body["status"] == "cancelled"
    assert body["cancel_requested_by"] == "pi-lead"
    assert body["cancel_confirm_source"] == "complete"
    assert body["terminal"] is True
