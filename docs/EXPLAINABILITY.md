# Explainability

Two layers, deliberately kept apart:

1. **Reason codes** on every score: rule-based *risk indicators* computed from the input
   features. Stable, auditable, understandable by a reviewer.
2. **Model attributions** in training reports: XGBoost gain importance (and optional
   permutation importance over the contract columns). These describe what the model relies
   on in aggregate.

Neither layer is a causal explanation of a single prediction. A transaction can score HIGH
with no reason code (the model found an interaction the rules do not cover) or LOW with
several (the model judged the indicators benign in context). The API documents reason
codes as indicators, and the README repeats the distinction.

## Reason codes (`fraud_ml.scoring.explanation`)

| Code | Fires when |
|---|---|
| `HIGH_AMOUNT_VS_HISTORY` | amount >= 4 x the account's 30-day median |
| `RAPID_OUTFLOW_AFTER_INFLOW` | prior 24h inflow > 0 and amount >= 50% of that inflow |
| `HIGH_VELOCITY_1H` | at least 3 prior transactions in the last hour |
| `MANY_COUNTERPARTIES_24H` | at least 5 distinct counterparties in the prior 24h |
| `NEW_BENEFICIARY` | counterparty not seen before on the account |
| `DORMANT_ACCOUNT_REACTIVATED` | no transaction for 30 days or more |
| `CROSS_BORDER` | beneficiary country differs from the transaction country |
| `NEAR_THRESHOLD_AMOUNT` | amount within 15% below 1,000 / 5,000 / 10,000 reference units |
| `HIGH_RISK_IP` | upstream IP risk band is HIGH |
| `NEW_DEVICE` | device not seen before on the account |
| `UNUSUAL_HOUR` | 00:00-05:59 UTC |
| `HIGH_VELOCITY_24H` | at least 10 prior transactions in the prior 24h |
| `NEW_ACCOUNT` | account younger than 30 days |

Codes are emitted in the order above (reviewer priority) and capped at five per score. They
are computed for every transaction regardless of its score, so a PASS decision with
indicators is visible to a monitoring process.

Example: the Docker smoke test scores `examples/transaction.json` (large transfer, new
beneficiary abroad, at night, new device, high-risk IP) as CRITICAL / PRIORITY_REVIEW with
`HIGH_AMOUNT_VS_HISTORY, NEW_BENEFICIARY, CROSS_BORDER, HIGH_RISK_IP, NEW_DEVICE`.

## Model attributions

* `feature_importance.csv`: gain (XGBoost) or |coefficient| (logistic regression) per
  transformed feature, with the share of total.
* `permutation_importance.csv` (opt-in, `--permutation-importance`): drop in average
  precision when each *contract column* is permuted on the validation split. This answers
  the data owner's question ("which inputs does the scorer depend on?") rather than the
  modeller's.

## Why not SHAP

SHAP would add a compiled dependency and per-request latency for an explanation that is
still local-linear and still not causal. For a decision-support demo, rule-based
indicators plus aggregate importance give reviewers something they can act on without
over-claiming. It is listed in the roadmap as a possible addition behind a flag.
