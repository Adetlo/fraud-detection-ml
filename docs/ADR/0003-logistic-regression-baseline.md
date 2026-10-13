# ADR 0003: A class-weighted logistic regression baseline is mandatory

**Status:** accepted (2026-10-13)

## Context

Tree ensembles are the default choice for tabular fraud data, but the added complexity
(hyper-parameters, early stopping, calibration distortion, opaque splits) must be justified
against a transparent model on the same features. Without a baseline, a PR-AUC of 0.76 has
no reference point.

## Decision

Every training run fits a logistic regression with median imputation, standard scaling,
one-hot encoding (shared pipeline), `class_weight="balanced"`, `C=0.5` and 2,000
iterations, and evaluates it through exactly the same calibration, threshold policy and
reporting as the advanced model. The baseline's test metrics are stored in the winning
artifact's metadata (`baseline_test_metrics`) whenever the advanced model wins.

## Consequences

* The demo shows why this matters: the baseline is within 0.003 test PR-AUC of XGBoost and
  wins outright at 10,000 rows. A reader sees the real size of the gain.
* `select_winner` can choose the baseline; the code and docs do not assume the advanced
  model wins.
* The baseline costs about half a second to fit on 50,000 rows, so there is no reason to
  skip it.
