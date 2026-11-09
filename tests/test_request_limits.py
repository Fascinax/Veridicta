import pytest
import ast
from pathlib import Path
from functools import wraps
from types import SimpleNamespace

from ui.request_limits import RequestLimits, RequestPolicy, RequestLimitExceeded


def test_concurrent_limit_releases_slot_after_error():
    limits = RequestLimits(RequestPolicy(max_concurrent=1))
    with limits.admit("one", "question"):
        with pytest.raises(RequestLimitExceeded):
            with limits.admit("two", "question"):
                pass
    with limits.admit("two", "question"):
        pass


def test_rate_limit_is_scoped_and_expires(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("ui.request_limits.time.monotonic", lambda: clock[0])
    limits = RequestLimits(RequestPolicy(requests_per_minute=1))
    with limits.admit("one", "question"):
        pass
    with pytest.raises(RequestLimitExceeded):
        with limits.admit("one", "again"):
            pass
    with limits.admit("two", "question"):
        pass
    clock[0] += 61
    with limits.admit("one", "question"):
        pass


def test_long_query_is_rejected_before_execution():
    limits = RequestLimits(RequestPolicy(max_query_chars=5))
    with pytest.raises(RequestLimitExceeded):
        with limits.admit("one", "too long"):
            pytest.fail("oversized question was admitted")


def test_ui_limit_wrapper_does_not_render_identity_widgets_again():
    source = Path("ui/app.py").read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "_bounded_request")
    limits = RequestLimits()
    namespace = {"wraps": wraps, "RequestLimitExceeded": RequestLimitExceeded,
                 "_get_request_limits": lambda: limits, "st": SimpleNamespace(session_state={})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "ui/app.py", "exec"), namespace)
    handler = namespace["_bounded_request"](lambda prompt, **kwargs: prompt)
    assert handler("question", workspace_store=SimpleNamespace(scope_key="owner")) == "question"
