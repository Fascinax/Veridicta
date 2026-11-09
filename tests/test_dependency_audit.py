from tools.dependency_audit import classify_findings


def test_exception_does_not_hide_a_new_advisory_or_package_version():
    findings = classify_findings(
        [("ragas", "0.4.3"), ("ragas", "0.4.4")],
        [{"vulns": [{"id": "new-advisory"}]}, {"vulns": [{"id": "GHSA-95ww-475f-pr4f"}]}],
    )
    assert all(row["status"] == "requires_triage" for row in findings)


def test_known_text_only_exception_remains_visible_in_report():
    findings = classify_findings([("ragas", "0.4.3")], [{"vulns": [{"id": "GHSA-95ww-475f-pr4f"}]}])
    assert findings[0]["status"] == "accepted_text_only_evaluation"
