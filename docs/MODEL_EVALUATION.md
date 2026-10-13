# Model evaluation

## Why accuracy is not reported

At a 1% fraud rate, a model that never flags anything is 99% accurate and worthless. Every
metric here is chosen for a rare-positive problem with an operational capacity constraint.

| Metric | Role |
|---|---|
| **PR-AUC** (average precision) | primary ranking metric (ADR 0002): the precision-recall trade-off over the whole score range, insensitive to the size of the negative class |
| ROC-AUC | reported, not used for selection. With 7,400 legitimate rows per 67 fraud rows, a few hundred false positives barely move the false-positive *rate*, so ROC-AUC sits above 0.99 for both models while precision at the operating point is 0.27-0.33. ROC looks optimistic under severe imbalance. |
| precision, recall, F1, false-positive rate | at the selected threshold |
| Brier score | probability quality (calibration + sharpness) |
| **review rate**, **alerts per 1,000** | operational load: what fraction of transactions would go to a reviewer |
| **expected illustrative cost** | `FN x 100 + FP x 5 + TP x 1` by default (synthetic units, configurable); drives the threshold choice together with the recall target |
| confusion matrix | at the selected threshold |
| segment performance | per currency, channel, transaction type and domestic vs cross-border |

## Protocol

1. Temporal split 70 / 15 / 15 (ADR 0001).
2. Fit the baseline on train. Fit XGBoost on train with early stopping on validation.
3. For each candidate: calibration decision on validation, threshold analysis and policy on
   validation, metrics on validation at its own threshold.
4. Score the test split once per candidate at that threshold.
5. Select the winner on **validation** numbers (PR-AUC first, with an operational override
   for a cheaper candidate that also needs fewer reviews, and a hard rule that a candidate
   which alone satisfies the policy wins).
6. Report both validation and test for both candidates. The test split is never used to
   choose anything.

## Demo results (50,000 rows, seed 42, generated 2026-10-13)

Dataset: 50,000 transactions, 501 fraud rows (1.00%), 1,250 accounts, 2025-01-01 to
2025-06-30. Split: train 35,000 (338 positives, 0.97%), validation 7,500 (96, 1.28%), test
7,500 (67, 0.89%).

| Metric | LR validation | LR test | XGBoost validation | XGBoost test |
|---|---:|---:|---:|---:|
| PR-AUC | 0.667 | 0.754 | **0.786** | 0.757 |
| ROC-AUC | 0.989 | 0.992 | 0.991 | 0.992 |
| precision | 0.335 | 0.272 | 0.360 | 0.326 |
| recall | 0.917 | 0.925 | 0.896 | 0.836 |
| F1 | 0.490 | 0.420 | 0.513 | 0.469 |
| Brier | 0.0067 | 0.0041 | 0.0052 | 0.0041 |
| review rate | 3.51% | 3.04% | 3.19% | 2.29% |
| alerts / 1,000 | 35.1 | 30.4 | 31.9 | 22.9 |
| expected cost | 1,763 | 1,392 | 1,851 | 1,736 |
| threshold | 0.051 | | 0.065 | |

Test confusion matrix for the selected XGBoost model at threshold 0.065: TP 56, FP 116,
FN 11, TN 7,317.

### Reading these numbers honestly

* XGBoost is selected: it leads validation PR-AUC by 0.12 and needs fewer reviews.
* On the test period the gap closes (0.757 vs 0.754) and the baseline reaches higher recall
  at a slightly higher review rate, which makes it *cheaper* under the illustrative cost
  model on that period. With 67 test positives, PR-AUC has a standard error of roughly
  ±0.05, so the two models are statistically indistinguishable out of sample on this data.
* The honest conclusion is that on this synthetic data, with these features, a
  class-weighted linear model is a strong baseline. That is a result worth knowing before
  shipping a tree ensemble, and it is why the baseline is mandatory (ADR 0003).
* At 10,000 rows (benchmark), the baseline wins outright: trees need more positives.
* Both models' ROC-AUC above 0.99 says little; the precision-recall curve
  (`reports/demo/precision_recall_curve.png`) is the picture that matters.

### Feature importance (XGBoost gain, top 10)

| Transformed feature | share of gain |
|---|---:|
| velocity_ratio_1h_24h | 9.6% |
| new_beneficiary_flag | 8.3% |
| new_device_flag | 4.5% |
| ip_risk_band_UNKNOWN | 4.0% |
| log1p_days_since_last | 3.9% |
| channel_ONLINE | 3.8% |
| near_threshold_flag | 2.7% |
| channel_CARD | 2.6% |
| cross_border_flag | 2.6% |
| ip_risk_band_MEDIUM | 2.5% |

Gain importance describes what the trees split on, not causal effect. Permutation importance
over raw contract columns is available with `fraud-ml train --permutation-importance`.

### Segment performance (test split, selected model)

`reports/demo/segment_metrics.csv`. Segments with fewer than 50 rows are omitted; ranking
metrics are left blank for segments without positives. Highlights: cross-border rows
(638 rows, 4.7% positive) reach recall 0.97 at a 7.1% review rate, domestic rows (6,862 rows,
0.54% positive) recall 0.73 at 1.9%. Channel ONLINE carries 51 of the 67 test positives.
This is **segment performance**, not a fairness analysis: no protected attributes exist in
the data.

## Calibration

Class weighting distorts probabilities, so calibration is measured on every run
(`calibration.csv`, `calibration_curve.png`, Brier score) and a calibrator is applied only
when it improves the validation Brier score by at least 2% without costing more than 0.005
PR-AUC. Platt scaling is preferred because it is monotone; isotonic regression is only
considered with at least 500 validation positives and must beat Platt by a further 10%.

On the demo run no calibration was applied: with square-root positive weighting the raw
XGBoost scores already have a validation Brier of 0.0052 and the reliability table on test
is close to the diagonal in the populated bins. The experiment that led to these rules is
summarised in ADR 0004: isotonic calibration on 96 positives consistently lost 0.03-0.08
test PR-AUC by flattening the ranking.

## Artifacts

`reports/` after `fraud-ml train` or `fraud-ml demo` (the committed demo copy is in
`reports/demo/`):

`metrics.json`, `threshold_analysis.csv`, `segment_metrics.csv`, `feature_importance.csv`,
`calibration.csv`, `precision_recall_curve.png`, `roc_curve.png`, `confusion_matrix.png`,
`calibration_curve.png`, `threshold_analysis.png`, `drift_report.json`.
