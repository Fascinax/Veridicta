"""Capture source-page evidence for empty LegiMonaco bodies, without repairing legal text."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup
import requests

from retrievers.snapshots import file_digest, verify_manifest


def inspect_page(html: bytes) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one(".documentContent #content")
    if content is None:
        raise ValueError("LegiMonaco source content container missing")
    banner = bool(content.select(".abrogatedText"))
    editorial_mention = any(re.search("abrog", note.get_text(" ", strip=True), re.I)
                            for note in content.select(".editoNotes"))
    for node in content.select("header, aside, nav, .histoLink, .histoList"):
        node.decompose()
    candidate_text = content.get_text(" ", strip=True)
    return {"source_html_sha256": hashlib.sha256(html).hexdigest(),
            "candidate_body_chars_after_metadata_exclusion": len(candidate_text),
            "html_repeal_banner": banner, "editorial_repeal_mention": editorial_mention,
            "classification": "no_body_after_metadata_exclusion" if not candidate_text else "body_present_review_extraction",
            "legal_review": "pending"}


def source_page_url(source: str) -> str:
    parts = urlsplit(source)
    if (parts.scheme != "https" or parts.netloc != "legimonaco.mc"
            or not parts.path.startswith("/tnc/") or parts.query or parts.fragment):
        raise ValueError("Expected an exact LegiMonaco text URL")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/", "", ""))


def review_empty_sources(manifest_path: Path, destination: Path, session=None) -> dict:
    manifest = verify_manifest(manifest_path)
    source_root = manifest_path.parent.resolve()
    destination = destination.resolve()
    if destination.is_relative_to(source_root) or source_root.is_relative_to(destination):
        raise ValueError("Source evidence must be separate from the snapshot")
    destination.mkdir(parents=True, exist_ok=False)
    report = {"source_manifest_sha256": file_digest(manifest_path), "checked_at": datetime.now(timezone.utc).isoformat(),
              "records": [], "legal_release_ready": False,
              "semantics": "HTML structure and publisher labels only; no legal status or text is repaired."}
    client = session or requests.Session()
    try:
        for relative in sorted(manifest["files"]):
            if not relative.startswith("raw/") or not relative.endswith(".jsonl"):
                continue
            with (source_root / relative).open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row.get("text"), str) or row["text"].strip():
                        continue
                    source = row.get("source", "")
                    parts = urlsplit(source)
                    if parts.scheme != "https" or parts.netloc != "legimonaco.mc" or not parts.path.startswith("/tnc/"):
                        continue
                    page_url = source_page_url(source)
                    response = client.get(page_url, headers={"User-Agent": "VeridictaCorpus/1.0"}, timeout=30, allow_redirects=False)
                    if response.status_code != 200:
                        raise ValueError("Source page must return HTTP 200 without redirect")
                    html = response.content
                    evidence = inspect_page(html)
                    filename = hashlib.sha256(source.encode()).hexdigest() + ".html"
                    (destination / filename).write_bytes(html)
                    evidence.update(id=row["id"], source=source, requested_url=page_url, api_abrogee=row.get("metadata", {}).get("abrogee"),
                                    html_filename=filename)
                    report["records"].append(evidence)
                    time.sleep(1.2)
    finally:
        if session is None:
            client.close()
    report["status"] = "review_evidence_collected"
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New, non-existing evidence directory")
    args = parser.parse_args()
    report = review_empty_sources(args.manifest, args.out)
    print(json.dumps({"status": report["status"], "reviewed_source_pages": len(report["records"]),
                      "legal_release_ready": False}))


if __name__ == "__main__":
    main()
