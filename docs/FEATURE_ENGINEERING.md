# Feature engineering

`FEATURE_VERSION = "features/1"`. The catalogue lives in `fraud_ml.features.definitions`
(`FEATURE_SPECS`), and the derivation in one pure function, `build_feature_frame`, that is
embedded in the persisted pipeline. 37 features in four groups feed the preprocessing step.

## Groups

**Transaction (12)**: `log_amount` (log1p of the amount in reference units), `amount_band`
(six buckets), `threshold_proximity` (amount / next round threshold among 1,000 / 5,000 /
10,000 reference units; 0 above the largest), `near_threshold_flag` (within 15% below a
threshold), and the categoricals `currency`, `transaction_type`, `channel`, `country`,
`beneficiary_country`, `merchant_category`, `ip_risk_band`, `customer_segment`.

Amounts are compared across currencies through static *illustrative* factors
(`CURRENCY_TO_REFERENCE`); they are not market FX rates and only serve to make amount
features comparable.

**Temporal (7)**: `hour_of_day`, `day_of_week`, `hour_sin`, `hour_cos`,
`unusual_hour_score` (1.0 for 00:00-04:59, 0.75 at 05:00, 0.5 at 22:00-23:59, 0.25 at
06:00 and 21:00, else 0), `is_weekend`, `is_night`. Derived from the row's own timestamp only.

**Velocity (8)**: `transactions_1h`, `transactions_24h`, `log1p_amount_1h`,
`log1p_amount_24h`, `unique_counterparties_24h`, `velocity_ratio_1h_24h`,
`amount_vs_inflow_24h` (amount / prior 24h inflow, 0 without inflow), `inflow_present_24h`.

**Behavioural (10)**: `amount_vs_account_median`, `amount_vs_account_mean` (missing without
30-day history), `log1p_days_since_last`, `log1p_account_age_days`, `has_history`,
`new_beneficiary_flag`, `new_device_flag`, `cross_border_flag`, `dormant_flag` (no
transaction for 30 days), `new_account_flag` (younger than 30 days).

Identifiers (`transaction_id`, `account_id`, `counterparty_id`, `device_id_hash`) are never
features.

## Preprocessing (the only fitted state)

| Column kind | Transformer |
|---|---|
| numeric | median imputation, then standard scaling for the logistic baseline (trees skip scaling) |
| binary | most-frequent imputation |
| closed categorical | one-hot, `handle_unknown="ignore"` |
| open categorical (`country`, `beneficiary_country`, `merchant_category`) | one-hot with `min_frequency=20` and `handle_unknown="infrequent_if_exist"` (explicit pooled bucket) |

The fitted `ColumnTransformer` outputs a pandas frame with named columns, so XGBoost gain
importance can be reported per transformed feature.

## Point-in-time history

`compute_history_features` is the reference computation of the contract's context fields.
For the transaction at position *i* within an account (ordered by timestamp, then id), every
value uses rows at positions `< i` only:

* window counts and sums use `searchsorted` on the timestamp array, so the window is
  `(t - w, t]` restricted to earlier positions;
* unique counterparties use a sliding multiset over the 24h window;
* 30-day median and mean are computed over the window's earlier rows (NaN when empty);
* `is_new_beneficiary` / `is_new_device` compare against everything seen earlier on the
  account;
* account age counts from `account_open_date` when given, else from the first observed
  transaction.

The generator uses this routine, and so does the `txn-data-pipeline` adapter. Online scoring
expects an upstream process to deliver the same fields; there is no streaming feature
computation in this project.

## Leakage controls

| Risk | Control | Test |
|---|---|---|
| Label enters the feature matrix | `FORBIDDEN_COLUMNS` includes `is_fraud`, `fraud_type`; `build_feature_frame` raises `LeakageError` when a label column is present in its input and asserts no forbidden column in its output | `tests/unit/test_leakage.py::test_target_column_cannot_enter_feature_matrix`, `test_pipeline_rejects_label_in_training_frame` |
| Post-investigation fields | `POST_OUTCOME_COLUMNS` (`investigation_outcome`, `confirmed_fraud`, `chargeback_flag`, `case_id`, `alert_id`, `analyst_decision`, `reported_at`, `risk_probability`, `fraud_score`, `decision_support`) are rejected on input | `test_post_outcome_fields_are_rejected` (parametrised) |
| Future transaction information | history features use strictly earlier rows; feature derivation is row-wise | `tests/unit/test_history.py::test_history_never_uses_future_rows`, property test `test_history_features_never_use_future_timestamps`, `test_future_rows_do_not_change_earlier_feature_rows` |
| Label-derived aggregates | no aggregate is computed from `is_fraud`; the generator derives history from the raw event log before labels are considered | `test_generator_is_reproducible_and_valid` (labels only on fraud rows) |
| Random split for final evaluation | `temporal_split` is the only split; validation and test periods follow training; assertion on boundaries | `tests/unit/test_split_synthetic.py`, `test_training_frames_do_not_overlap_in_time` |
| Preprocessing fitted on validation/test | the preprocessor is fitted on the training split only; test asserts learned categories and scaler statistics come from the training split | `test_encoders_are_fitted_on_training_split_only`, `test_scaler_statistics_come_from_training_split` |
| Threshold or calibrator tuned on test | both are selected on the validation split; the test split is scored once | `tests/e2e/test_full_flow.py` (threshold equals the validation decision) |
| Identifiers as features | `ID_COLUMNS` are forbidden | `test_identifiers_and_raw_timestamp_are_not_features` |

### Account-level overlap

The same accounts appear in every split, as in production where a model scores future
transactions of known accounts. This is not leakage because no row carries information from
later rows. What *would* be leakage is an account-level statistic computed over the whole
dataset (e.g. the account's overall fraud rate or lifetime median); none exists here. A
purge gap between splits is not needed for the same reason, and is noted in ADR 0001 as an
option if window features were ever computed with look-ahead.

### Known cold-start effect

Because the synthetic window starts on day one for every account, early transactions have
no history: `is_new_beneficiary` is true for the first use of every counterparty and
`amount_vs_account_median` is missing. The drift check on the demo flags
`is_new_beneficiary` as a WARNING between the training period and the latest period for
exactly this reason. It is a property of the synthetic history, documented rather than hidden.
