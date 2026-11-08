import ast
import json
from pathlib import Path

import pytest

from tools.access_check import check_access_configuration, main
from workspaces.access import (
    AccessDenied, AccessPolicy, AccessPolicyError, MAX_POLICY_BYTES,
    POLICY_ENV, authorized_workspace_owner,
)
from workspaces.identity import workspace_owner

ISSUER = "https://idp.example.invalid/tenant-a"
CLAIMS = {"is_logged_in": True, "iss": ISSUER, "sub": "synthetic-one", "exp": 4_000_000_000}


def _write_policy(path, principals=None):
    if principals is None:
        principals = [{"iss": ISSUER, "sub": CLAIMS["sub"]}]
    path.write_text(json.dumps({"schema_version": 1, "principals": principals}), encoding="utf-8")
    return path


@pytest.fixture
def policy_file(tmp_path, monkeypatch):
    path = _write_policy(tmp_path / "policy.json")
    monkeypatch.setenv(POLICY_ENV, str(path))
    monkeypatch.setenv("VERIDICTA_WORKSPACE_MODE", "authenticated")
    monkeypatch.setenv("VERIDICTA_WORKSPACE_DIR", str(tmp_path / "private-store"))
    return path


def test_approved_identity_keeps_existing_owner_id(policy_file):
    assert authorized_workspace_owner("authenticated", "session", CLAIMS) == workspace_owner("authenticated", "session", CLAIMS)


@pytest.mark.parametrize("changes", [
    {"sub": "unapproved"}, {"iss": "https://idp.example.invalid/tenant-b"},
    {"iss": ISSUER + "/"}, {"sub": "SYNTHETIC-ONE"},
    {"sub": "unapproved", "email": "synthetic-one@approved.invalid", "email_verified": True},
])
def test_only_exact_issuer_and_subject_grant_access(policy_file, changes):
    with pytest.raises(AccessDenied):
        authorized_workspace_owner("authenticated", "session", CLAIMS | changes)


def test_multiple_approved_pairs_do_not_authorize_their_cross_product(policy_file):
    other_issuer = "https://idp.example.invalid/tenant-b"
    _write_policy(policy_file, [{"iss": ISSUER, "sub": "synthetic-one"}, {"iss": other_issuer, "sub": "synthetic-two"}])
    authorized_workspace_owner("authenticated", "session", CLAIMS | {"iss": other_issuer, "sub": "synthetic-two"})
    with pytest.raises(AccessDenied):
        authorized_workspace_owner("authenticated", "session", CLAIMS | {"sub": "synthetic-two"})


def test_wildcard_strings_do_not_grant_wildcard_access(policy_file):
    _write_policy(policy_file, [{"iss": ISSUER, "sub": "*"}])
    with pytest.raises(AccessDenied):
        authorized_workspace_owner("authenticated", "session", CLAIMS)


@pytest.mark.parametrize("expiry", [None, "4000000000", True, 0, -1, float("nan"), float("inf"), 10**400])
def test_invalid_expiry_is_rejected(policy_file, expiry):
    with pytest.raises(PermissionError):
        authorized_workspace_owner("authenticated", "session", CLAIMS | {"exp": expiry})


def test_missing_expiry_is_rejected(policy_file):
    with pytest.raises(PermissionError):
        authorized_workspace_owner("authenticated", "session", {key: value for key, value in CLAIMS.items() if key != "exp"})


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_expiry_boundary_is_checked_each_time(policy_file, monkeypatch, offset):
    monkeypatch.setattr("workspaces.identity.time.time", lambda: 1000)
    claims = CLAIMS | {"exp": 1000 + offset}
    if offset <= 0:
        with pytest.raises(PermissionError):
            authorized_workspace_owner("authenticated", "session", claims)
    else:
        assert authorized_workspace_owner("authenticated", "session", claims).startswith("oidc:")


@pytest.mark.parametrize("not_before", [1001, None, True, "999", float("nan"), float("inf")])
def test_not_before_is_validated_when_present(policy_file, monkeypatch, not_before):
    monkeypatch.setattr("workspaces.identity.time.time", lambda: 1000)
    with pytest.raises(PermissionError):
        authorized_workspace_owner("authenticated", "session", CLAIMS | {"nbf": not_before})


def test_not_before_boundary_is_inclusive(policy_file, monkeypatch):
    monkeypatch.setattr("workspaces.identity.time.time", lambda: 1000)
    assert authorized_workspace_owner("authenticated", "session", CLAIMS | {"nbf": 1000}).startswith("oidc:")


def test_revocation_is_reloaded_within_the_same_session(policy_file):
    authorized_workspace_owner("authenticated", "same-session", CLAIMS)
    _write_policy(policy_file, [])
    with pytest.raises(AccessDenied):
        authorized_workspace_owner("authenticated", "same-session", CLAIMS)


def test_disappearing_policy_never_reuses_previous_approval(policy_file):
    authorized_workspace_owner("authenticated", "same-session", CLAIMS)
    policy_file.unlink()
    with pytest.raises(AccessPolicyError):
        authorized_workspace_owner("authenticated", "same-session", CLAIMS)


@pytest.mark.parametrize("filename", [None, "", " ", "missing-policy.json"])
def test_missing_policy_is_a_configuration_failure(filename):
    with pytest.raises(AccessPolicyError):
        AccessPolicy.load(filename)


@pytest.mark.parametrize("document", [
    [], {}, {"schema_version": True, "principals": []}, {"schema_version": 2, "principals": []},
    {"schema_version": 1, "principals": {}},
    {"schema_version": 1, "principals": [], "allow_everyone": True},
    {"schema_version": 1, "principals": [{"iss": ISSUER}]},
    {"schema_version": 1, "principals": [{"iss": ISSUER, "sub": ""}]},
    {"schema_version": 1, "principals": [{"iss": ISSUER, "sub": " padded "}]},
    {"schema_version": 1, "principals": [{"iss": ISSUER, "sub": ["one"]}]},
    {"schema_version": 1, "principals": [{"iss": ISSUER, "sub": "one", "role": "admin"}]},
    {"schema_version": 1, "principals": [{"iss": ISSUER, "sub": "one"}] * 2},
])
def test_malformed_policy_is_rejected(tmp_path, document):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(AccessPolicyError):
        AccessPolicy.load(str(path))


@pytest.mark.parametrize("payload", [
    b'{"schema_version":1,"principals":[],"principals":[]}',
    b'{"schema_version":1,"principals":[{"iss":"a","sub":"b","sub":"c"}]}',
    b"not json", b"\xff", b"[" * 1100, b" " * (MAX_POLICY_BYTES + 1),
])
def test_unreadable_or_ambiguous_policy_is_rejected(tmp_path, payload):
    path = tmp_path / "policy.json"
    path.write_bytes(payload)
    with pytest.raises(AccessPolicyError):
        AccessPolicy.load(str(path))


@pytest.mark.parametrize("mode, expected", [("session", "session:server-id"), ("local", None)])
def test_local_modes_do_not_require_an_oidc_policy(monkeypatch, mode, expected):
    monkeypatch.delenv(POLICY_ENV, raising=False)
    assert authorized_workspace_owner(mode, "server-id", {}) == expected


def _run_gate(claims, initial_state=None):
    from streamlit.testing.v1 import AppTest

    script = (
        "import streamlit as st\nfrom unittest.mock import patch\nimport ui.access as access\n"
        f"with patch.object(access.st, 'user', {claims!r}), "
        "patch.object(st, 'login', lambda: st.text('login-called')), "
        "patch.object(st, 'logout', lambda: st.text('logout-called')):\n"
        "    store = access.get_workspace_store()\n"
        "    st.text(str(len(store.list())))\n"
    )
    app = AppTest.from_string(script)
    if initial_state:
        for key, value in initial_state.items():
            app.session_state[key] = value
    return app.run()


@pytest.mark.parametrize("claims, error_count", [({}, 0), (CLAIMS | {"sub": "denied"}, 1), (CLAIMS | {"exp": 0}, 0)])
def test_gate_stops_before_storage_and_clears_private_state(policy_file, claims, error_count):
    app = _run_gate(claims, {"messages": ["private"], "note_old": "private", "_chip_query": "private"})
    assert not app.exception
    assert len(app.error) == error_count and not app.text
    assert not (policy_file.parent / "private-store").exists()
    assert not {"messages", "note_old", "_chip_query"} & app.session_state.keys()


def test_gate_configuration_error_does_not_echo_private_identifiers(policy_file):
    policy_file.write_text("private-invalid-json", encoding="utf-8")
    app = _run_gate(CLAIMS)
    assert not app.exception and len(app.error) == 1 and not app.text
    assert "private-invalid-json" not in app.error[0].value and ISSUER not in app.error[0].value


def test_gate_opens_an_authorized_scope_and_clears_previous_owner_widgets(policy_file):
    app = _run_gate(CLAIMS, {"workspace_owner_scope": "previous", "note_old": "private", "messages": ["private"]})
    assert not app.exception and app.text[0].value == "0"
    assert "note_old" not in app.session_state and "messages" not in app.session_state


def test_gate_rechecks_approval_on_rerun(policy_file):
    app = _run_gate(CLAIMS, {"messages": ["private"]})
    _write_policy(policy_file, [])
    app.run()
    assert not app.exception and len(app.error) == 1 and not app.text
    assert "messages" not in app.session_state


@pytest.mark.parametrize("claims, label, expected", [
    ({}, "Se connecter", "login-called"),
    (CLAIMS | {"exp": 0}, "Se reconnecter", "logout-called"),
    (CLAIMS | {"sub": "denied"}, "Se déconnecter", "logout-called"),
    (CLAIMS, "Se déconnecter", "logout-called"),
])
def test_auth_buttons_stop_before_private_body_even_if_auth_call_returns(policy_file, claims, label, expected):
    app = _run_gate(claims)
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception and [element.value for element in app.text] == [expected]


def test_application_stops_before_artifacts_sidebar_and_rag_for_denied_identity(policy_file):
    from streamlit.testing.v1 import AppTest

    source = Path("ui/app.py").read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "main")
    script = (
        "import streamlit as st\nimport os\nfrom unittest.mock import patch\n"
        "from ui.access import get_workspace_store as _get_workspace_store\n"
        + ast.get_source_segment(source, node)
        + f"\nwith patch.object(st, 'user', {CLAIMS | {'sub': 'denied'}!r}):\n    main()\n"
        + "st.text('Private application body reached')\n"
    )
    app = AppTest.from_string(script).run()
    assert not app.exception and len(app.error) == 1 and not app.text
    assert not (policy_file.parent / "private-store").exists()


@pytest.mark.parametrize("mode", ["session", "local", "unknown"])
def test_preflight_refuses_modes_unsuitable_for_private_deployment(mode):
    assert check_access_configuration({"VERIDICTA_WORKSPACE_MODE": mode})["configuration_valid"] is False


def test_preflight_refuses_empty_policy_and_does_not_expose_principals(policy_file):
    _write_policy(policy_file, [])
    report = check_access_configuration({"VERIDICTA_WORKSPACE_MODE": "authenticated", POLICY_ENV: str(policy_file)})
    assert report["reason"] == "no_authorized_principals" and report["configuration_valid"] is False


def test_preflight_success_is_not_a_live_oidc_validation(policy_file, capsys):
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {"configuration_valid": True, "principal_count": 1, "oidc_connection_verified": False, "deployment_validated": False}
    assert ISSUER not in json.dumps(report) and CLAIMS["sub"] not in json.dumps(report)


def test_preflight_failure_returns_nonzero_without_private_path(monkeypatch, capsys):
    monkeypatch.setenv("VERIDICTA_WORKSPACE_MODE", "authenticated")
    monkeypatch.setenv(POLICY_ENV, "private-missing-policy.json")
    assert main() == 1
    output = capsys.readouterr().out
    assert "private-missing-policy" not in output and json.loads(output)["configuration_valid"] is False
