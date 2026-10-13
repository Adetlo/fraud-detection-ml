# ADR 0004: XGBoost as the advanced model, with square-root positive weighting and guarded calibration

**Status:** accepted (2026-10-13)

## Context

One stronger model was wanted, chosen for dependency reliability and portability rather
than leaderboard performance. XGBoost and LightGBM both ship wheels for Linux, macOS and
Windows and both need an OpenMP runtime on macOS. XGBoost's scikit-learn wrapper supports
early stopping, `scale_pos_weight` and gain importance directly, and its `hist` method
trains 50,000 rows in about one second.

Class imbalance can be handled with the full negative/positive ratio as `scale_pos_weight`
(about 100 here). An experiment on two seeds showed that the full ratio maximises recall
but inflates probabilities (validation Brier 0.009-0.012 vs 0.004-0.005) and does not
improve PR-AUC, while no weighting under-trains on the positives at small sizes. The square
root of the ratio (about 10) kept ranking quality with usable probabilities. The same
experiment showed isotonic calibration fitted on 96 validation positives lowering test
PR-AUC by 0.03-0.08 (step function overfitting the ranking), while Platt scaling never hurt.

## Decision

* `XGBClassifier` with `tree_method="hist"`, depth 3, learning rate 0.05, subsample and
  column subsample 0.8, `min_child_weight` 5, up to 600 trees with early stopping (40
  rounds, `aucpr`) on the validation split; `scale_pos_weight = sqrt(neg/pos)` by default
  (`balanced` and `none` remain configurable).
* Calibration is measured every run; a calibrator is kept only if it improves validation
  Brier by at least 2% without losing more than 0.005 PR-AUC. Platt scaling is preferred;
  isotonic requires at least 500 validation positives and must beat Platt by a further 10%.
* LightGBM is not added; one advanced model is enough for a rigorous comparison.

## Consequences

* On the demo, no calibration was applied: the sqrt-weighted raw scores already had a
  validation Brier of 0.0052.
* Thresholds live on a calibrated-looking but still low scale (0.065 for a 1% base rate);
  the docs explain why that is expected.
* macOS developers need `libomp` (Homebrew) or an equivalent; CI and Docker install
  `libgomp` from the system.
