# Demo artifacts

Produced by `uv run fraud-ml demo --transactions 50000 --fraud-rate 0.01 --seed 42` on
2026-10-13 with the committed code, and copied from `reports/`. The README, the model card
and `docs/MODEL_EVALUATION.md` quote these files; regenerate them before changing any number.

| File | Content |
|---|---|
| `metrics.json` | full training result: both candidates (validation + test), threshold decisions, calibration decisions, split summary, selection rationale, artifact metadata |
| `threshold_analysis.csv` | confusion counts, precision, recall, review rate and expected cost for every candidate threshold (validation split, selected model) |
| `segment_metrics.csv` | test-split performance per currency, channel, transaction type and domestic/cross-border |
| `feature_importance.csv` | XGBoost gain per transformed feature |
| `calibration.csv` | reliability table (10 quantile bins, test split) |
| `precision_recall_curve.png`, `roc_curve.png` | both models on the test split |
| `confusion_matrix.png` | selected model at the selected threshold |
| `calibration_curve.png`, `threshold_analysis.png` | selected model |
| `drift_report.json` | training period vs latest period, including score drift |
