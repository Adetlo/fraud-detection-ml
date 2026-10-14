# History reconstruction

This repository was originally developed over an extended period beginning in 2024 and was later
consolidated and published to GitHub. The Git history was reconstructed from the final project
structure and known development phases. Historical commit dates represent approximate development
phases rather than forensic original commit timestamps.

## Reconstructed phases

| Phase | Approximate period | What it added |
|---|---|---|
| Project setup, contracts and data | Sep – Oct 2024 | pyproject/README/LICENSE scaffold, configuration and structured logging, `fraud-transaction/1` contract and enums, frame loader and validation, point-in-time account history features, synthetic generator, temporal split |
| Feature pipeline | Nov – Dec 2024 | feature catalogue with forbidden/post-outcome columns, transaction, temporal and behavioural features, the shared leakage-guarded pipeline |
| Baseline and calibration | Feb 2025 | class-weighted logistic regression baseline, guarded probability calibration |
| Advanced model and evaluation | May 2025 | `fraud-score/1` output contract, XGBoost with `scale_pos_weight`, metrics / segment performance / importance / plots |
| Threshold policy and explainability | Aug 2025 | threshold selection as an operational policy, rule-based reason codes, threshold/evaluation tests |
| Monitoring and MLOps | Oct – Nov 2025 | PSI drift checks, data-quality and score-distribution summaries, hash-verified serialization and artifact registry, local MLflow tracking, end-to-end training orchestration |
| Serving, CLI and test suites | Feb 2026 | scorer and batch scoring, FastAPI service, `fraud-ml` CLI, `txn-data-pipeline` adapter, shared fixtures and the remaining unit / API / integration / property / e2e suites |
| Containerisation and CI | May 2026 | Dockerfile, entrypoint with bootstrap model, docker-compose, first GitHub Actions workflow |
| Documentation | Aug – Sep 2026 | feature engineering, thresholding, drift, explainability and security docs, examples, benchmark script, expanded README |
| Publication | Oct 2026 | ADRs, architecture / data-contract / evaluation / governance / performance docs, model card, demo reports and sample data, community health files, release 1.0.0 with lockfile, refreshed CI |

## What is and is not authentic

* **File contents are authentic.** Every file in the final tree is byte-identical to the published
  state of the project. No source code or test was rewritten, and no file exists in the history
  that is not in the published repository (other than this document).
* **Commit dates, messages and grouping are reconstructed.** They describe the files each commit
  contains and follow the dependency order of the modules, but they are not the original commit
  timestamps. The exact original development dates are unknown.
* **The repository was published to GitHub in October 2026.** The pre-reconstruction history was
  a set of 14 commits all dated 2026-10-13.
* Known anachronisms, kept deliberately because files are committed once at their final content:
  * Intermediate versions exist only for `pyproject.toml`, `README.md`, `Dockerfile`,
    `.dockerignore` and `.github/workflows/ci.yml`. The final `pyproject.toml` (release commit)
    carries dependency lower bounds and the final CI workflow carries action versions that
    postdate the early commits; the intermediates use bounds and versions available at the time.
  * `LICENSE` is committed in September 2024 with the copyright year 2024 and updated to 2026 (the publication year) in the October 2026 community-files commit.
  * `src/fraud_ml/config.py` is committed in the first phase at its final content and already
    declares settings for MLflow, the API and the cost model, which arrive in later phases.
    `src/fraud_ml/data/synthetic.py` imports `features/velocity.py`, so the history-feature
    module precedes the rest of the feature pipeline.
  * Several unit tests (`test_split_synthetic.py`, `test_features.py`, `test_contracts.py`,
    `test_threshold_evaluation.py`) are committed with the feature they test but use fixtures
    from `tests/conftest.py`, which is committed once, at its final content, in February 2026
    (it imports the training pipeline and the scorer). Those tests would not have been runnable
    at their commit; this is a reconstruction artefact.
  * `src/fraud_ml/training/train.py` imports the registry, experiment tracker, evaluation and
    threshold modules, so the end-to-end training entry point is committed after all of them
    (November 2025) rather than alongside the first model.
  * ADR acceptance dates (2026-10-13) reflect when the decisions were written up, not when they
    were taken. `docs/ARCHITECTURE.md`, `MODEL_CARD.md`, `docs/MODEL_EVALUATION.md`,
    `docs/PERFORMANCE.md`, `reports/demo/` and `data/samples/` describe or derive from the demo
    run executed on 2026-10-13 and are therefore committed in the publication phase. The sample
    Parquet files carry synthetic 2025 transaction timestamps set by the generator configuration.
  * `uv.lock` resolves package versions released in 2026 and is committed in the release commit.
* No tags were created.

## Archive

The pre-reconstruction history (14 commits dated 2026-10-13) is preserved in an archive
branch/tag alongside this reconstructed history.
