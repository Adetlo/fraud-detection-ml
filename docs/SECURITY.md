# Security notes

This is an educational project on synthetic data. The controls below exist to show where
the trust boundaries of an ML scoring service are, not to claim production hardening.

## Data

* All data is synthetic. Identifiers use obvious synthetic prefixes (`TXN-SYN-`, `ACC-SYN-`,
  `CP-SYN-`, `DEV-SYN-`). No real customer, account or merchant data was used or is needed.
* Feature payloads are not logged by default. Logs carry identifiers, bands, decisions,
  timings and correlation ids.
* Validation error details sent to clients contain field locations and messages, not the
  submitted values.

## API

* **No authentication in v1.** A real deployment would sit behind OAuth2/OIDC (client
  credentials for service-to-service calls), mTLS inside the perimeter, and rate limiting.
  No fake bank authentication is implemented, on purpose.
* Input validation through the Pydantic contract: types, ranges, closed vocabularies,
  cross-field consistency, timezone-aware timestamps.
* Request-size limit (`FRAUD_ML_API_MAX_REQUEST_BYTES`, default 2 MB -> 413
  `REQUEST_TOO_LARGE`) and batch-size limit (`FRAUD_ML_API_MAX_BATCH_SIZE`, default 1,000 ->
  413 `BATCH_TOO_LARGE`).
* Structured errors with stable codes (`INVALID_INPUT`, `UNSUPPORTED_CATEGORY`,
  `SCHEMA_VERSION_MISMATCH`, `MODEL_NOT_READY`, `MODEL_VERSION_MISMATCH`, `SCORING_FAILED`,
  `INTERNAL_ERROR`) and a correlation id. Stack traces are logged, never returned.
* `/api/v1/model` exposes versions, policy and metrics, never filesystem paths or secrets.
* Docker image runs as a non-root user; the API binds to `127.0.0.1:8000` in compose.

## Model artifacts are code

`joblib`/`pickle` files execute arbitrary code on load. A model artifact must therefore be
treated as **code from a trusted build**, not as data:

* the registry loads only local artifact directories, never files supplied by API clients;
* `metadata.json` records the SHA-256 of `model.joblib` at training time, and the loader
  refuses to unpickle when the on-disk hash differs (`ARTIFACT_INTEGRITY`);
* schema, feature and score-contract versions must match the running code
  (`MODEL_VERSION_MISMATCH`), so a stale artifact cannot silently score with the wrong
  feature semantics;
* readiness is false until a verified artifact is loaded.

A production system would sign artifacts (e.g. Sigstore) and verify the signature in
addition to the hash, store them in an access-controlled registry, and build the image from
the same pinned environment that produced them.

## Dependencies and static analysis

* `uv.lock` pins every dependency; CI runs `pip-audit` on the exported lock and `bandit`
  on `src`. Two bandit findings are reviewed and marked: the `subprocess` call that reads
  the git commit with a fixed argv, and the string `"PASS"` (a decision label, not a password).
* No secrets exist in the repository; `.env.example` holds defaults only.

## Reporting

See the root [SECURITY.md](../SECURITY.md).
