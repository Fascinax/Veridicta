"""Check local access configuration without connecting to an identity provider."""

from collections.abc import Mapping
import json
import os

from workspaces.access import AccessPolicy, AccessPolicyError, POLICY_ENV


def check_access_configuration(settings: Mapping[str, str]) -> dict:
    report = {
        "configuration_valid": False,
        "oidc_connection_verified": False,
        "deployment_validated": False,
    }
    if settings.get("VERIDICTA_WORKSPACE_MODE", "session") != "authenticated":
        return report | {"reason": "authenticated_mode_required"}
    try:
        policy = AccessPolicy.load(settings.get(POLICY_ENV))
    except AccessPolicyError:
        return report | {"reason": "missing_or_invalid_access_policy"}
    if not policy.principals:
        return report | {"reason": "no_authorized_principals", "principal_count": 0}
    return report | {"configuration_valid": True, "principal_count": len(policy.principals)}


def main() -> int:
    report = check_access_configuration(os.environ)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["configuration_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
