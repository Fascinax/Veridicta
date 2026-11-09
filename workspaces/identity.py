"""Resolve server-trusted identities without using browser-supplied IDs."""

from collections.abc import Mapping
import json
import math
import time


def workspace_owner(mode: str, session_id: str, claims: Mapping | None = None) -> str | None:
    if mode == "local":
        return None
    if mode == "session":
        if not session_id:
            raise ValueError("session identity is required")
        return "session:" + session_id
    if mode != "authenticated":
        raise ValueError("unknown workspace mode")
    user = claims or {}
    issuer, subject = user.get("iss"), user.get("sub")
    if user.get("is_logged_in") is not True or not isinstance(issuer, str) or not isinstance(subject, str):
        raise PermissionError("authenticated issuer and subject are required")
    if not issuer.strip() or not subject.strip():
        raise PermissionError("authenticated issuer and subject are required")
    now = time.time()
    expiry = _numeric_date(user.get("exp"))
    if expiry <= now:
        raise PermissionError("identity expired")
    if "nbf" in user and _numeric_date(user["nbf"]) > now:
        raise PermissionError("identity not yet valid")
    return "oidc:" + json.dumps([issuer, subject], separators=(",", ":"))


def _numeric_date(value: object) -> float:
    if type(value) not in (int, float):
        raise PermissionError("invalid identity time")
    try:
        timestamp = float(value)
    except OverflowError:
        raise PermissionError("invalid identity time") from None
    if not math.isfinite(timestamp):
        raise PermissionError("invalid identity time")
    return timestamp
