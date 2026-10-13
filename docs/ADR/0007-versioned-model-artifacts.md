# ADR 0007: Versioned, hash-verified model artifacts with compatibility gates

**Status:** accepted (2026-10-13)

## Context

A scoring service that loads "whatever is in the models directory" cannot say which data,
code, features or policy produced a score, and a pickle file is executable code. Model
versioning must cover lineage, compatibility and integrity, without a registry server.

## Decision

An artifact is a directory with `model.joblib` (the whole scikit-learn pipeline, including
feature derivation, preprocessing, model and any calibrator) and `metadata.json`
(`ModelMetadata`): model name/version/type, run id, schema/feature/score-contract versions,
training timestamp, training data hash and name, row count, git commit, seed, library
versions, training configuration, feature names, risk policy, calibration method,
validation and test metrics, baseline metrics, selection rationale, split summary and the
SHA-256 of `model.joblib`.

`load_artifact` refuses to load when metadata is missing or invalid, when any contract
version differs from the running code, when the hash is absent or does not match, or when
the object lacks `predict_proba`. `latest` is an explicit promotion (a copy), one directory
per run is kept, and the API reports the loaded version on `/health/ready` and
`/api/v1/model`.

## Consequences

* Rollback is copying a previous run to `latest`; policy rolls back with the model.
* Changing the feature catalogue requires bumping `FEATURE_VERSION`, which invalidates old
  artifacts on purpose.
* A full model registry (stages, approvals, signatures) is out of scope; the metadata is
  shaped so it could be pushed to one later.
