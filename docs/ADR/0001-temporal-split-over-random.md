# ADR 0001: Temporal split over random split

**Status:** accepted (2026-10-13)

## Context

Financial transactions are a time series of account behaviour. A random train/test split
places a transaction from June in the training set and its account's May transactions in
the test set: the model is evaluated on the past of accounts whose future it has already
seen, and seasonal or drift effects are averaged away. Reported metrics are then optimistic
in a way that only shows up after deployment.

## Decision

`fraud_ml.data.split.temporal_split` is the only splitting function. Rows are ordered by
`transaction_timestamp` (ties by `transaction_id`) and cut at row fractions 70 / 15 / 15.
Validation strictly follows training; test strictly follows validation. Early stopping,
calibration and threshold selection use the validation period only. The split boundaries
and per-period positive rates are stored in the artifact metadata.

Accounts are *not* separated across splits. The deployed model will score future
transactions of known accounts, so account overlap is the realistic condition. Leakage is
prevented at the feature level instead: every history feature uses strictly earlier rows
and no dataset-wide account statistic exists.

## Consequences

* Positive counts per period are small (96 validation, 67 test on the demo) and the
  period-to-period fraud rate varies (0.97% / 1.28% / 0.89%); metrics carry visible
  uncertainty and the docs say so.
* No purge gap is needed because window features never look ahead; if a feature with
  look-ahead were introduced, a gap of at least the window length would be required and
  the splitter has a natural place to add it.
* Random K-fold cross-validation is deliberately not offered for model selection.
