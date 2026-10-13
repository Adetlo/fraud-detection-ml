# ADR 0005: The threshold is an operational policy selected on validation

**Status:** accepted (2026-10-13)

## Context

A fixed 0.5 threshold encodes two hidden assumptions, equal error costs and unlimited
review capacity, and with class weighting it is not even the centre of the probability
scale. Fraud operations care about three things at once: how much fraud is caught
(recall), how many reviews it costs (review rate / alerts per 1,000) and what the misses
and wasted reviews cost (expected cost).

## Decision

Threshold selection is a configurable policy (`ThresholdPolicy`): reach a recall target
within a review-rate ceiling and, among feasible thresholds, minimise an explicit,
illustrative cost model (`CostModel`). The sweep over candidate thresholds is vectorised,
written to `threshold_analysis.csv`, and performed on the validation split only. A second,
higher priority threshold (first point reaching a precision target) separates REVIEW from
PRIORITY_REVIEW. Risk bands are defined relative to these thresholds and persisted with the
artifact, so a score response always carries the threshold that produced its decision.

If the policy cannot be satisfied, the run says so (`policy_satisfied=false`) and falls
back to the best achievable recall within capacity rather than failing silently.

## Consequences

* Changing business constraints (recall target, capacity, costs) is a configuration
  change and a retraining run, not a code change.
* Cost values are clearly labelled as illustrative units; nothing in the repository claims
  real loss figures.
* Thresholds are not comparable across models or runs; the artifact is the unit of
  deployment, model plus policy together.
