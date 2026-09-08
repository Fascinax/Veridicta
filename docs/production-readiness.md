# Production readiness checklist

This checklist is the operational gate for the Veridicta RAG demo and API
paths. It separates repository-verifiable controls from deployment controls
that require a live provider, corpus or hosting environment.

## Before a release

- [x] Run `python -m eval.validate_contract --strict-human-labels`.
- [x] Run `python -m eval.validate_consistency`.
- [x] Run `pytest tests -m "not slow and not benchmark"`.
- [x] Run the source compilation check from `.github/workflows/ci.yml`.
- [x] Normalize the selected raw snapshot and retain its manifest. The
  manifest records input/output digests, source filename, record count and
  UTC refresh time.
- [ ] Run the readiness gate against the manifest used by the deployment:

  ```bash
  python -m retrievers.readiness \
    --manifest data/raw/<snapshot>.jsonl.manifest.json \
    --source-count <retrieved-source-count> \
    --max-age-days 30
  ```

- [ ] Perform one provider-backed smoke request with synthetic, non-sensitive
  input and record the provider, model, latency and failure behavior.
- [ ] Confirm the hosting secret store contains the required credentials and
  that `.env`, tokens and provider responses are not part of the artifact.

## Failure behavior

| Boundary | Expected behavior | Repository check |
| --- | --- | --- |
| No retrieved sources | Abstain without calling a generation provider. | `tests/test_production_readiness.py` |
| Provider timeout | Normalize to a safe timeout error; do not expose the original message. | `tests/test_omniroute_client.py` |
| Stale or incomplete corpus manifest | Fail the readiness gate before release. | `tests/test_production_readiness.py` |
| Missing required index artifact | Fail startup with an actionable artifact error. | `tests/test_artifacts.py` |
| Optional index artifact | Warn and keep the documented local fallback. | `tests/test_artifacts.py` |
| Unsupported claims or invalid citations | Remove them and append a source limitation. | `tests/test_grounding.py` |

## Security and operations

- Audit traces are metadata-only by default; full source content requires the
  explicit `VERIDICTA_AUDIT_INCLUDE_CONTENT` opt-in.
- UI generation errors use generic, user-safe messages. Provider exception
  text, credentials and source content are not rendered in the demo.
- The CI workflow has read-only repository permissions and does not need
  provider credentials.
- Corpus and index artifacts must be versioned outside Git when they are large
  or private; retain the manifest digest beside the deployed artifact.
- Tenant isolation, authenticated deletion/export workflows and a production
  retention policy remain deployment work. They are not inferred from the
  local demo and must be completed before handling private multi-tenant data.

## Release evidence

Attach the CI run, readiness JSON, corpus manifest digest, provider smoke
result and the exact source revision to the release record. A green local test
suite alone does not prove provider availability, hosting isolation or legal
correctness.
