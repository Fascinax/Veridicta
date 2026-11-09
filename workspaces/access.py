"""Authorize verified OIDC claims using a server-managed list of principals."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path

from workspaces.identity import workspace_owner

POLICY_ENV = "VERIDICTA_ACCESS_POLICY_FILE"
MAX_POLICY_BYTES = 64 * 1024


class AccessPolicyError(ValueError):
    """The server cannot establish its authorization policy."""


class AccessDenied(PermissionError):
    """A valid identity is absent from the approved principals."""


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise AccessPolicyError("duplicate policy key")
        result[key] = value
    return result


def _principal(value: object) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) != {"iss", "sub"}:
        raise AccessPolicyError("invalid principal fields")
    if any(not isinstance(item, str) or not item or item != item.strip() for item in value.values()):
        raise AccessPolicyError("invalid principal identifiers")
    return value["iss"], value["sub"]


@dataclass(frozen=True, repr=False)
class AccessPolicy:
    principals: frozenset[tuple[str, str]]

    @classmethod
    def load(cls, filename: str | None) -> "AccessPolicy":
        if not filename or not filename.strip():
            raise AccessPolicyError("access policy file is required")
        try:
            with Path(filename).open("rb") as stream:
                payload = stream.read(MAX_POLICY_BYTES + 1)
            if len(payload) > MAX_POLICY_BYTES:
                raise AccessPolicyError("access policy exceeds size limit")
            document = json.loads(payload, object_pairs_hook=_unique_object)
        except (OSError, ValueError, RecursionError):
            raise AccessPolicyError("access policy is unreadable or invalid") from None
        if not isinstance(document, dict) or set(document) != {"schema_version", "principals"}:
            raise AccessPolicyError("invalid access policy fields")
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise AccessPolicyError("unsupported access policy version")
        if not isinstance(document["principals"], list):
            raise AccessPolicyError("principals must be a list")
        principals = [_principal(value) for value in document["principals"]]
        if len(set(principals)) != len(principals):
            raise AccessPolicyError("duplicate principal")
        return cls(frozenset(principals))

    def require(self, claims: Mapping) -> None:
        if (claims.get("iss"), claims.get("sub")) not in self.principals:
            raise AccessDenied("identity is not authorized")


def authorized_workspace_owner(mode: str, session_id: str, claims: Mapping) -> str | None:
    if mode != "authenticated":
        return workspace_owner(mode, session_id)
    # Reload for each rerun: revocation must not depend on a cached login result.
    policy = AccessPolicy.load(os.getenv(POLICY_ENV))
    owner = workspace_owner(mode, session_id, claims)
    policy.require(claims)
    return owner
