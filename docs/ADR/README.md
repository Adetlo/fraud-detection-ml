# Architecture decision records

| ADR | Decision |
|---|---|
| [0001](0001-temporal-split-over-random.md) | Temporal split over random split for all evaluation |
| [0002](0002-pr-auc-primary-ranking-metric.md) | PR-AUC as the primary ranking metric |
| [0003](0003-logistic-regression-baseline.md) | A class-weighted logistic regression baseline is mandatory |
| [0004](0004-gradient-boosting-primary-model.md) | XGBoost as the advanced model, with square-root positive weighting and guarded calibration |
| [0005](0005-threshold-as-operational-policy.md) | The threshold is an operational policy selected on validation |
| [0006](0006-synthetic-data-only.md) | Synthetic data only, independent of the sibling repositories |
| [0007](0007-versioned-model-artifacts.md) | Versioned, hash-verified model artifacts with compatibility gates |

Format: context, decision, consequences. Status is *accepted* unless stated.
