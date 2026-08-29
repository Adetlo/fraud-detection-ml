# Threshold selection and risk bands

## The threshold is a policy, not a constant

A probability threshold of 0.5 assumes that a false negative and a false positive cost the
same and that reviewers have unlimited capacity. Neither is true in fraud operations, and
with class weighting the probability scale is not centred on 0.5 anyway. The threshold is
therefore selected by an explicit, configurable policy (ADR 0005) on the **validation**
split and persisted in the model artifact.

## Policy (`fraud_ml.config.ThresholdPolicy`)

| Setting | Default | Meaning |
|---|---|---|
| `recall_target` | 0.80 | catch at least 80% of synthetic fraud |
| `max_review_rate` | 0.05 | send at most 5% of transactions to review |
| `priority_precision_target` | 0.75 | precision at which a score becomes PRIORITY_REVIEW |
| `low_band_fraction` | 0.5 | LOW/MEDIUM boundary as a fraction of the threshold |

Environment variables: `FRAUD_ML_RECALL_TARGET`, `FRAUD_ML_MAX_REVIEW_RATE`,
`FRAUD_ML_PRIORITY_PRECISION_TARGET`.

### Algorithm (`fraud_ml.training.threshold`)

1. Build a candidate grid from the score quantiles plus a uniform grid (so sparse regions
   are covered) and compute the confusion counts for every candidate in one vectorised pass.
2. Keep candidates with `review_rate <= max_review_rate` and `recall >= recall_target`.
3. Among them choose the lowest **expected illustrative cost**; ties go to fewer alerts,
   then to the higher threshold.
4. If nothing satisfies both constraints, mark `policy_satisfied = false` and use the
   best-recall candidate within the review-rate ceiling. The flag is stored in the training
   result and printed, so an unsatisfied policy is never silent.
5. `priority_threshold`: the first threshold above the selected one where precision reaches
   `priority_precision_target`, or the midpoint to 1.0 if none exists.
6. `low_upper = threshold x low_band_fraction`.

`threshold_analysis.csv` holds the full sweep (threshold, TP/FP/TN/FN, precision, recall,
F1, FPR, review rate, alerts per 1,000, expected cost) and `threshold_analysis.png` plots
precision, recall and review rate against the threshold with the selection marked.

## Cost model (`fraud_ml.config.CostModel`)

| Outcome | Default cost | Env var |
|---|---:|---|
| false negative (missed fraud) | 100 | `FRAUD_ML_FALSE_NEGATIVE_COST` |
| false positive (wasted review) | 5 | `FRAUD_ML_FALSE_POSITIVE_COST` |
| true positive (investigation) | 1 | `FRAUD_ML_TRUE_POSITIVE_COST` |
| true negative | 0 | |

`expected_cost = FN x 100 + FP x 5 + TP x 1`. The values are **illustrative synthetic units**.
They are not estimates of real financial loss and should be replaced with an institution's
own loss and operations figures before the policy means anything. `expected_cost_per_1000`
normalises across datasets.

## Review capacity

`review_rate` is the share of transactions at or above the threshold and
`alerts_per_1000 = 1000 x review_rate`. A model with 95% recall at a 40% review rate is
operationally useless; the ceiling makes that trade visible and enforceable.

## Risk bands and decision support (`fraud_ml.contracts.score.RiskPolicy`)

| Band | Probability range | Decision support |
|---|---|---|
| LOW | `p < low_upper` | PASS |
| MEDIUM | `low_upper <= p < threshold` | PASS |
| HIGH | `threshold <= p < priority_threshold` | REVIEW |
| CRITICAL | `p >= priority_threshold` | PRIORITY_REVIEW |

There is no BLOCK decision. The output prioritises human or downstream-control attention;
it does not decide. Band boundaries travel with the artifact (`metadata.json -> policy`) and
are returned by `GET /api/v1/model`.

## Demo values (seed 42, 50k rows)

| | selected XGBoost |
|---|---:|
| threshold | 0.065 |
| priority threshold | 0.41 |
| LOW upper bound | 0.0325 |
| validation recall / review rate | 0.896 / 3.19% |
| test recall / review rate | 0.836 / 2.29% |
| policy satisfied | yes |

A threshold of 0.065 looks low only because the positive rate is 1%: a transaction with a
6.5% fraud probability is six times riskier than average.
