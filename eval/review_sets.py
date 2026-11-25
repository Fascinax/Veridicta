"""Validate independent-case intake and refuse unreviewed legal quality claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def question_key(text: str) -> str:
    normalized = "".join(character for character in unicodedata.normalize("NFKD", text).casefold()
                         if not unicodedata.combining(character))
    return " ".join(re.findall(r"\w+", normalized))


def validate_intake(cases: list[dict], regression: list[dict], *, require_reviewed: bool = False) -> dict:
    if not isinstance(cases, list) or not all(isinstance(row, dict) for row in cases):
        raise ValueError("Cases must be a list of objects")
    if not isinstance(regression, list) or not all(isinstance(row, dict) and isinstance(row.get("question"), str) for row in regression):
        raise ValueError("Regression cases must provide questions")
    existing = {question_key(row["question"]) for row in regression}
    ids, questions = set(), set()
    reviewed = 0
    for row in cases:
        if not isinstance(row.get("id"), str) or not row["id"].strip() or not isinstance(row.get("question"), str):
            raise ValueError("Case IDs and questions must be nonempty strings")
        key = question_key(row["question"])
        if not key or row["id"] in ids or key in questions or key in existing:
            raise ValueError("Case IDs and questions must be unique and absent from the regression set")
        ids.add(row["id"])
        questions.add(key)
        if row.get("review_status") == "reviewed":
            if not all(isinstance(row.get(field), str) and row[field].strip() for field in ("reviewed_by", "reference_answer")):
                raise ValueError("Reviewed legal cases require an explicit reviewer, answer and passages")
            passages = row.get("supporting_passages")
            if not isinstance(passages, list) or not passages or not all(
                isinstance(passage, dict) and all(isinstance(passage.get(field), str) and passage[field].strip()
                                                  for field in ("source_url", "text")) for passage in passages
            ):
                raise ValueError("Evidence requires source URLs and exact passages")
            reviewed += 1
        elif row.get("review_status") != "pending":
            raise ValueError("Unknown review status")
    if require_reviewed and (not cases or reviewed != len(cases)):
        raise ValueError("Legal evaluation blocked: human review is incomplete")
    return {"cases": len(cases), "reviewed": reviewed, "pending": len(cases) - reviewed, "eligible_for_legal_evaluation": bool(cases) and reviewed == len(cases), "scope": "review_intake_not_a_sealed_holdout"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("eval/datasets/legal_review_intake.json"))
    parser.add_argument("--regression", type=Path, default=Path("eval/test_questions.json"))
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    report = validate_intake(json.loads(raw), json.loads(args.regression.read_text(encoding="utf-8")), require_reviewed=args.require_reviewed)
    report["sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
