# Drift and monitoring

## Four kinds of drift

| Kind | What changes | Needs labels | Implemented |
|---|---|---|---|
| **Data drift** | distribution of inputs (amounts, velocities, channels) | no | yes: PSI per input column |
| **Score drift** | distribution of model outputs; review rate | no | yes: PSI on `risk_probability` |
| **Performance drift** | precision / recall / cost at the operating point | yes (delayed) | no: reported once at training time; `fraud-ml evaluate` can re-run it on a newly labelled file |
| **Concept drift** | the relationship between inputs and fraud | yes | no |

Only label-free checks are implemented, because labels in fraud arrive late, partially and
through a process this project does not have (no analyst feedback loop). The limitation is
stated rather than papered over.

## Method (`fraud_ml.monitoring.drift`)

Population Stability Index between a reference sample (training period) and a current sample:

```
PSI = sum_i (current_i - reference_i) x ln(current_i / reference_i)
```

* Numeric columns: 10 reference-quantile bins (a constant reference gets one bin around the
  constant and one on each side).
* Categorical columns: one bin per category (top 20 by reference frequency, rest pooled);
  the report also lists the largest frequency shifts and categories unseen in the reference.
* Scores: same numeric PSI on `risk_probability` when `--model` is given.

| PSI | Status |
|---|---|
| < 0.10 | OK |
| 0.10 - 0.25 | WARNING |
| >= 0.25 | DRIFT |

Report status: OK when every feature is OK; WARNING when any feature warns or drifts; DRIFT
when at least 25% of checked features drift or the score distribution drifts. Thresholds are
configurable (`DriftThresholds`).

Columns checked: 11 numeric (`amount`, `account_age_days`, `transactions_1h`,
`transactions_24h`, `amount_1h`, `amount_24h`, `unique_counterparties_24h`,
`days_since_last_transaction`, `account_median_amount_30d`, `account_mean_amount_30d`,
`inflow_amount_24h`) and 10 categorical (`currency`, `transaction_type`, `channel`,
`country`, `beneficiary_country`, `merchant_category`, `ip_risk_band`, `customer_segment`,
`is_new_beneficiary`, `is_new_device`).

## Report shape

```json
{
  "status": "WARNING",
  "features_checked": 21,
  "features_warning": 1,
  "features_drifted": 0,
  "drifted_features": [],
  "warning_features": ["is_new_beneficiary"],
  "score_drift": {"feature": "risk_probability", "psi": 0.0768, "status": "OK", "...": "..."}
}
```

The full report (`fraud-ml drift-check --output`) adds a per-feature entry with PSI, status,
row counts and details (means, missing rates, frequency shifts, new categories).

The demo compares the training period with the latest 15% of the synthetic window. The
`is_new_beneficiary` warning is a real property of the data: early in the window every
counterparty is new (cold start), later most are known. See
[FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md).

## Other monitoring outputs

* `fraud_ml.monitoring.score_distribution.summarise_scores`: mean, quantiles, band rates,
  review rate, alerts per 1,000 and a histogram of a batch of scores.
* `fraud_ml.monitoring.data_quality.data_quality_report`: invalid-row rate with error
  codes, missing rates for required fields, numeric ranges.
* Structured logs: every API request logs method, path, status and `duration_ms` with the
  correlation id; every score logs `model_version`, `transaction_id`, band and decision.
  Feature payloads are never logged.

## What a real deployment would add

Performance drift on delayed labels, alert outcome rates from analyst feedback, per-segment
drift, population sampling with time windows, and alert routing. None of it is here.
