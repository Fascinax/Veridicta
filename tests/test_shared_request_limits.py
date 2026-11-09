import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import hashlib
import multiprocessing
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from filelock import SoftFileLock
import pytest

import ui.shared_request_limits as shared
from ui.request_limits import RequestLimitExceeded, RequestPolicy
from ui.shared_request_limits import RequestLimitsUnavailable, SharedRequestLimits, request_limits_from_environment


def _hold_worker(database, policy, control):
    ready, release = control
    with SharedRequestLimits(database, policy).admit("synthetic-holder", "synthetic request"):
        ready.put("holding")
        release.wait(30)


def _rate_worker(database, policy, control):
    barrier, results = control
    limits = SharedRequestLimits(database, policy)
    barrier.wait(timeout=30)
    try:
        with limits.admit("synthetic-owner", "synthetic request"):
            results.put("admitted")
    except RequestLimitsUnavailable:
        results.put("unavailable")
    except RequestLimitExceeded:
        results.put("rate_rejected")


@pytest.fixture
def held_worker(tmp_path):
    context = multiprocessing.get_context("spawn")
    database = tmp_path / "quota.sqlite3"
    policy = RequestPolicy(max_concurrent=1)
    limits = SharedRequestLimits(database, policy)
    ready, release = context.Queue(), context.Event()
    process = context.Process(target=_hold_worker, args=(database, policy, (ready, release)))
    process.start()
    try:
        assert ready.get(timeout=30) == "holding"
        yield limits, process
    finally:
        if process.is_alive():
            release.set()
            process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        ready.close()


def test_concurrency_limit_is_shared_with_another_process(held_worker):
    limits, _ = held_worker
    with pytest.raises(RequestLimitExceeded):
        with limits.admit("other-owner", "synthetic request"):
            pytest.fail("another process already holds the only slot")


def test_killed_worker_releases_its_slot_without_an_expiring_lease(held_worker):
    limits, process = held_worker
    process.terminate()
    process.join(timeout=10)
    assert not process.is_alive()
    with limits.admit("other-owner", "synthetic request"):
        pass


def test_racing_processes_share_one_owner_rate_limit(tmp_path):
    context = multiprocessing.get_context("spawn")
    database = tmp_path / "quota.sqlite3"
    policy = RequestPolicy(max_concurrent=6, requests_per_minute=3)
    SharedRequestLimits(database, policy)
    barrier, results = context.Barrier(6), context.Queue()
    processes = [context.Process(target=_rate_worker, args=(database, policy, (barrier, results))) for _ in range(6)]
    try:
        for process in processes:
            process.start()
        statuses = [results.get(timeout=40) for _ in processes]
        for process in processes:
            process.join(timeout=10)
        assert Counter(statuses) == {"admitted": 3, "rate_rejected": 3}
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        results.close()


def test_independent_instances_and_restart_keep_the_rate_counter(tmp_path):
    path = tmp_path / "quota.sqlite3"
    policy = RequestPolicy(requests_per_minute=1)
    with SharedRequestLimits(path, policy).admit("owner", "one"):
        pass
    with pytest.raises(RequestLimitExceeded):
        with SharedRequestLimits(path, policy).admit("owner", "two"):
            pytest.fail("a new instance must not reset the counter")


def test_rate_window_expires_at_its_boundary(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(shared.time, "time", lambda: now[0])
    limits = SharedRequestLimits(tmp_path / "quota.sqlite3", RequestPolicy(requests_per_minute=1))
    with limits.admit("one", "one"):
        pass
    now[0] = 1059.99
    with pytest.raises(RequestLimitExceeded):
        with limits.admit("one", "two"):
            pass
    now[0] = 1060
    with limits.admit("one", "three"):
        pass


def test_existing_owner_can_continue_when_owner_capacity_is_full(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "MAX_OWNER_BUCKETS", 1)
    limits = SharedRequestLimits(tmp_path / "quota.sqlite3")
    with limits.admit("existing", "one"):
        pass
    with pytest.raises(RequestLimitExceeded):
        with limits.admit("new", "two"):
            pass
    with limits.admit("existing", "three"):
        pass


def test_full_owner_capacity_can_be_reclaimed_after_expiry(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(shared, "MAX_OWNER_BUCKETS", 1)
    monkeypatch.setattr(shared.time, "time", lambda: now[0])
    limits = SharedRequestLimits(tmp_path / "quota.sqlite3")
    with limits.admit("old", "one"):
        pass
    now[0] += 60
    with limits.admit("new", "two"):
        pass
    with sqlite3.connect(limits.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM owners").fetchone()[0] == 1


def test_clock_rollback_does_not_evict_an_owner_with_live_events(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(shared, "MAX_OWNER_BUCKETS", 1)
    monkeypatch.setattr(shared.time, "time", lambda: now[0])
    limits = SharedRequestLimits(tmp_path / "quota.sqlite3")
    with limits.admit("existing", "one"):
        pass
    now[0] = 990.0
    with limits.admit("existing", "two"):
        pass
    now[0] = 1050.0
    with pytest.raises(RequestLimitExceeded):
        with limits.admit("new", "three"):
            pass


def test_provider_error_releases_slot_and_keeps_admitted_attempt_in_quota(tmp_path):
    limits = SharedRequestLimits(tmp_path / "quota.sqlite3", RequestPolicy(max_concurrent=1, requests_per_minute=1))
    with pytest.raises(RuntimeError, match="provider fixture"):
        with limits.admit("one", "one"):
            raise RuntimeError("provider fixture")
    with limits.admit("two", "two"):
        pass
    with pytest.raises(RequestLimitExceeded):
        with limits.admit("one", "retry"):
            pass


def test_query_rejection_does_not_consume_rate_or_concurrency(tmp_path):
    limits = SharedRequestLimits(tmp_path / "quota.sqlite3", RequestPolicy(max_query_chars=3, requests_per_minute=1))
    with pytest.raises(RequestLimitExceeded):
        with limits.admit("owner", "too long"):
            pass
    with limits.admit("owner", "ok"):
        pass


def test_database_contains_no_raw_owner_or_query(tmp_path):
    limits = SharedRequestLimits(tmp_path / "quota.sqlite3")
    with limits.admit("sensitive-owner-fixture", "sensitive-question-fixture"):
        pass
    content = limits.database.read_bytes()
    assert b"sensitive-owner-fixture" not in content and b"sensitive-question-fixture" not in content
    with sqlite3.connect(limits.database) as connection:
        owner, = connection.execute("SELECT owner FROM owners").fetchone()
    assert owner == hashlib.sha256(b"sensitive-owner-fixture").hexdigest()


def test_policy_mismatch_cannot_silently_create_a_second_quota(tmp_path):
    path = tmp_path / "quota.sqlite3"
    SharedRequestLimits(path)
    with pytest.raises(RequestLimitsUnavailable):
        SharedRequestLimits(path, RequestPolicy(max_concurrent=3))


def test_foreign_database_is_rejected_without_changing_its_bytes(tmp_path):
    path = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE private_data (value TEXT)")
        connection.execute("INSERT INTO private_data VALUES ('synthetic secret')")
    before = path.read_bytes()
    with pytest.raises(RequestLimitsUnavailable):
        SharedRequestLimits(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("payload", [b"not a sqlite file", b"\xff" * 128])
def test_corrupted_database_fails_closed_without_recreation(tmp_path, payload):
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(payload)
    with pytest.raises(RequestLimitsUnavailable):
        SharedRequestLimits(path)
    assert path.read_bytes() == payload


def test_sqlite_lock_contention_fails_closed_and_releases_the_slot(tmp_path, monkeypatch):
    limits = SharedRequestLimits(tmp_path / "quota.sqlite3", RequestPolicy(max_concurrent=1))
    monkeypatch.setattr(shared, "BUSY_TIMEOUT_SECONDS", 0.05)
    with sqlite3.connect(limits.database, isolation_level=None) as connection:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(RequestLimitsUnavailable, match="quotas est indisponible"):
            with limits.admit("one", "one"):
                pytest.fail("locked quota database must stop execution")
        connection.rollback()
    with limits.admit("two", "two"):
        pass


def test_soft_lock_fallback_is_refused(tmp_path, monkeypatch):
    limits = SharedRequestLimits(tmp_path / "quota.sqlite3")
    monkeypatch.setattr(shared, "FileLock", SoftFileLock)
    with pytest.raises(RequestLimitsUnavailable):
        with limits.admit("owner", "one"):
            pass


def test_wal_database_is_not_used_or_silently_converted(tmp_path):
    path = tmp_path / "quota.sqlite3"
    SharedRequestLimits(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    with pytest.raises(RequestLimitsUnavailable):
        SharedRequestLimits(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_threads_share_the_same_concurrency_slots(tmp_path):
    limits = SharedRequestLimits(tmp_path / "quota.sqlite3", RequestPolicy(max_concurrent=1))
    def attempt():
        with pytest.raises(RequestLimitExceeded):
            with limits.admit("other", "two"):
                pass
    with limits.admit("holder", "one"):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(attempt).result(timeout=10)


def test_default_factory_places_quota_next_to_workspace_storage(tmp_path, monkeypatch):
    monkeypatch.delenv("VERIDICTA_REQUEST_LIMITS_DB", raising=False)
    monkeypatch.setenv("VERIDICTA_WORKSPACE_DIR", str(tmp_path))
    assert request_limits_from_environment().database == tmp_path / ".request-limits.sqlite3"


def test_factory_uses_explicit_shared_database_path(tmp_path, monkeypatch):
    path = tmp_path / "shared.sqlite3"
    monkeypatch.setenv("VERIDICTA_REQUEST_LIMITS_DB", str(path))
    assert request_limits_from_environment().database == path


def test_ui_stops_before_handler_if_shared_quota_is_unavailable(tmp_path):
    source = Path("ui/app.py").read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "_bounded_request")
    path = tmp_path / "private-missing-parent" / "foreign.sqlite3"
    path.parent.mkdir()
    path.write_bytes(b"not a sqlite database")
    warnings, executed = [], []
    namespace = {"wraps": wraps, "RequestLimitExceeded": RequestLimitExceeded,
                 "_get_request_limits": lambda: SharedRequestLimits(path),
                 "st": SimpleNamespace(session_state={}, warning=warnings.append)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "ui/app.py", "exec"), namespace)
    handler = namespace["_bounded_request"](lambda prompt, **kwargs: executed.append(prompt))
    handler("sensitive-question-fixture", workspace_store=SimpleNamespace(scope_key="sensitive-owner-fixture"))
    assert not executed and warnings == [shared.UNAVAILABLE_MESSAGE]
    assert str(path) not in warnings[0]
