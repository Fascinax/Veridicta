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
    if "exp" in user:
        try:
            expiry = float(user["exp"])
            if not math.isfinite(expiry) or expiry <= time.time():
                raise PermissionError("identity expired")
        except (ValueError, TypeError):
            raise PermissionError("invalid identity expiry") from None
    return "oidc:" + json.dumps([issuer, subject], separators=(",", ":"))
