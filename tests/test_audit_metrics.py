import json

from tools.audit_metrics import summarize_audits


def test_metrics_count_errors_and_corrupt_rows_without_returning_content(tmp_path):
    rows = [{"latency_s": 1, "query": "private", "stage_latencies_s": {"retrieval": .2}},
            {"latency_s": 3, "failure_classification": {"generation_error": True}}]
    (tmp_path / "traces.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\ninvalid\n", encoding="utf-8")
    result = summarize_audits(tmp_path)
    assert result["request_count"] == 2 and result["invalid_row_count"] == 1 and result["generation_error_count"] == 1
    assert result["latency_s"]["median"] == 2 and "private" not in json.dumps(result)
