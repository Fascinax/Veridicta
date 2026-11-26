"""Audit installed versions against OSV, retaining narrowly scoped exceptions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import tomllib
from pathlib import Path
from urllib.request import Request, urlopen


# Neither vulnerable path is used by the text-only evaluator. Exceptions must
# disappear when upgrading the package or enabling multimodal/untrusted caches.
EXCEPTIONS = {
    ("ragas", "0.4.3"): {"GHSA-95ww-475f-pr4f", "PYSEC-2026-3046"},
    ("diskcache", "5.6.3"): {"GHSA-w8v5-vhqr-4h9v", "PYSEC-2026-2447"},
}


def evaluation_only_packages(lock: dict) -> set[str]:
    """Use the union of locked platform dependencies when granting exceptions."""
    edges = defaultdict(set)
    for row in lock["package"]:
        edges[row["name"]].update(dep["name"] for dep in row.get("dependencies", []))
        for extra in row.get("optional-dependencies", {}).values():
            edges[row["name"]].update(dep["name"] for dep in extra)
    project = next(row for row in lock["package"] if row.get("source") == {"virtual": "."})
    groups = project.get("dev-dependencies", {})

    def reachable(roots):
        pending = list(roots)
        visited = set()
        while pending:
            name = pending.pop()
            if name not in visited:
                visited.add(name)
                pending.extend(edges[name])
        return visited

    evaluation = reachable(dep["name"] for dep in groups.get("evaluation", []))
    other = {project["name"]}
    other.update(dep["name"] for group, deps in groups.items() if group != "evaluation" for dep in deps)
    return evaluation - reachable(other)


def classify_findings(packages: list[tuple[str, str]], results: list[dict], *, evaluation_only=frozenset()) -> list[dict]:
    if len(packages) != len(results):
        raise ValueError("OSV response does not match the package inventory")
    findings = []
    for (name, version), result in zip(packages, results):
        for advisory in result.get("vulns", []):
            advisory_id = advisory["id"]
            findings.append({
                "package": name,
                "version": version,
                "advisory": advisory_id,
                "status": "accepted_text_only_evaluation" if name in evaluation_only and advisory_id in EXCEPTIONS.get((name.lower(), version), set()) else "requires_triage",
            })
    return findings


def audit_packages(packages: list[tuple[str, str]], *, evaluation_only=frozenset()) -> dict:
    payload = {"queries": [{"package": {"name": name, "ecosystem": "PyPI"}, "version": version.partition("+")[0]} for name, version in packages]}
    request = Request("https://api.osv.dev/v1/querybatch", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        results = json.load(response)["results"]
    findings = classify_findings(packages, results, evaluation_only=evaluation_only)
    return {"packages_checked": len(packages), "findings": findings, "unresolved": sum(row["status"] == "requires_triage" for row in findings)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lockfile", type=Path)
    args = parser.parse_args()
    evaluation_only = set()
    if args.lockfile:
        lock = tomllib.loads(args.lockfile.read_text(encoding="utf-8"))
        evaluation_only = evaluation_only_packages(lock)
        for row in lock["package"]:
            if "registry" not in row["source"] and row["source"] != {"virtual": "."}:
                raise ValueError(f"Unsupported dependency source for {row['name']}")
        packages = sorted({(row["name"], row["version"]) for row in lock["package"] if "registry" in row.get("source", {})})
    else:
        packages = sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata.get("Name")})
    report = audit_packages(packages, evaluation_only=evaluation_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dependency audit: {report['packages_checked']} packages, {report['unresolved']} unresolved advisories")
    raise SystemExit(1 if report["unresolved"] else 0)


if __name__ == "__main__":
    main()
