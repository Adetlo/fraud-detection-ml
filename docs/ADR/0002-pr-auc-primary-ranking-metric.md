# ADR 0002: PR-AUC as the primary ranking metric

**Status:** accepted (2026-10-13)

## Context

With a 1% positive rate, accuracy is meaningless (99% for "never fraud") and ROC-AUC is
optimistic: its x-axis, the false-positive *rate*, is diluted by the huge negative class, so
a model that produces hundreds of false alerts per true alert still scores above 0.99. The
quantity that matters operationally is precision at achievable recall, i.e. how many
reviews are wasted per fraud caught.

## Decision

Average precision (area under the precision-recall curve) is the primary metric for model
comparison and for tracking in MLflow. ROC-AUC is reported for completeness with an
explicit caveat in the docs and the plot title. Final model selection additionally
considers the operational point: expected illustrative cost and review rate at the
policy-selected threshold, and whether the policy could be satisfied at all.

## Consequences

* The demo shows the effect: both models have ROC-AUC 0.99 while precision at the
  operating point is 0.27-0.33.
* PR-AUC on 67-96 positives has roughly ±0.05 uncertainty, so small differences are
  reported as ties rather than wins.
* Threshold-free comparison (PR-AUC) and threshold-dependent operations (cost, review
  rate) are both visible, and the selection rule between them is written down in
  `select_winner`.
