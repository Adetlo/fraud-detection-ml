"""``fraud-ml`` command-line interface."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from fraud_ml import __version__
from fraud_ml.config import Settings, get_settings
from fraud_ml.exceptions import FraudMLError
from fraud_ml.logging import configure_logging

app = typer.Typer(
    name="fraud-ml",
    help="Fraud risk scoring and decision support for synthetic financial transactions (educational).",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()
err_console = Console(stderr=True)

DEFAULT_DATA = Path("data/generated/transactions.parquet")


def _settings(ctx: typer.Context) -> Settings:
    settings: Settings = ctx.obj["settings"]
    return settings


def _fail(exc: FraudMLError) -> None:
    err_console.print(f"[red]error[/red] {exc.code}: {exc.message}")
    raise typer.Exit(code=1)


@app.callback()
def main(
    ctx: typer.Context,
    log_level: Annotated[
        str | None, typer.Option("--log-level", help="DEBUG, INFO, WARNING, ERROR")
    ] = None,
    log_format: Annotated[str | None, typer.Option("--log-format", help="console or json")] = None,
) -> None:
    settings = get_settings()
    overrides: dict[str, Any] = {}
    if log_level:
        overrides["log_level"] = log_level
    if log_format:
        overrides["log_format"] = log_format
    if overrides:
        settings = settings.model_copy(update=overrides)
    configure_logging(settings.log_level, settings.log_format)
    ctx.obj = {"settings": settings}


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)


@app.command("generate-data")
def generate_data(
    transactions: Annotated[int, typer.Option("--transactions", "-n", min=100)] = 50_000,
    fraud_rate: Annotated[float, typer.Option("--fraud-rate", min=0.0, max=0.5)] = 0.01,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    days: Annotated[int, typer.Option("--days", min=7)] = 180,
    start: Annotated[str, typer.Option("--start", help="First day (YYYY-MM-DD)")] = "2025-01-01",
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory or .parquet/.jsonl/.csv file")
    ] = Path("data/generated/"),
) -> None:
    """Generate a reproducible synthetic labelled dataset."""
    from fraud_ml.data.loader import write_frame
    from fraud_ml.data.synthetic import GeneratorConfig, generate_dataset

    config = GeneratorConfig(
        transactions=transactions,
        fraud_rate=fraud_rate,
        seed=seed,
        days=days,
        start=date.fromisoformat(start),
    )
    frame, summary = generate_dataset(config)
    target = output / "transactions.parquet" if output.suffix == "" else output
    write_frame(frame, target)
    (target.parent / "generation_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    console.print(
        f"[green]wrote[/green] {target} ({summary.transactions:,} rows, fraud rate {summary.fraud_rate:.2%}, {summary.accounts:,} accounts)"
    )
    console.print(f"fraud types: {summary.fraud_type_counts}")


def _training_config(
    settings: Settings, seed: int, model_version: str, fast: bool, permutation: bool
) -> Any:
    from fraud_ml.models.gradient_boosting import GradientBoostingParams
    from fraud_ml.training.train import TrainingConfig

    boosting = (
        GradientBoostingParams(n_estimators=120, early_stopping_rounds=15)
        if fast
        else GradientBoostingParams()
    )
    return TrainingConfig(
        seed=seed,
        model_version=model_version,
        gradient_boosting=boosting,
        cost_model=settings.cost_model,
        threshold_policy=settings.threshold_policy,
        permutation_importance=permutation,
    )


def _print_training(result: Any) -> None:
    table = Table(title=f"Training run {result.run_id} - winner: {result.winner}")
    table.add_column("metric")
    for name in result.candidates:
        table.add_column(f"{name} (val)", justify="right")
        table.add_column(f"{name} (test)", justify="right")
    rows = (
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "review_rate",
        "alerts_per_1000",
        "expected_cost",
    )
    for metric in rows:
        values = []
        for candidate in result.candidates.values():
            for split in ("validation_metrics", "test_metrics"):
                value = getattr(getattr(candidate, split), metric)
                values.append(
                    f"{value:.4f}"
                    if isinstance(value, float)
                    and metric not in ("expected_cost", "alerts_per_1000")
                    else f"{value:.1f}"
                )
        table.add_row(metric, *values)
    console.print(table)
    winning = result.winning
    console.print(
        f"selected threshold: {winning.threshold.threshold:.4f} (priority {winning.threshold.priority_threshold:.4f}); policy satisfied: {winning.threshold.policy_satisfied}"
    )
    console.print(f"calibration: {winning.calibration.rationale}")
    console.print(f"decision: {result.selection_rationale}")
    console.print(
        f"artifact: {result.artifact_path}  reports: {result.reports_path}  ({result.total_seconds:.1f}s)"
    )


@app.command()
def train(
    ctx: typer.Context,
    input_path: Annotated[Path, typer.Option("--input", "-i")] = DEFAULT_DATA,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Model root (default: settings.model_dir)")
    ] = None,
    reports: Annotated[
        Path | None,
        typer.Option("--reports", help="Reports directory (default: settings.reports_dir)"),
    ] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    model_version: Annotated[str, typer.Option("--model-version")] = "fraud-risk/1.0.0",
    fast: Annotated[bool, typer.Option("--fast", help="Fewer boosting rounds (CI)")] = False,
    permutation: Annotated[bool, typer.Option("--permutation-importance")] = False,
    no_mlflow: Annotated[bool, typer.Option("--no-mlflow")] = False,
) -> None:
    """Train baseline + gradient boosting, select threshold, save a versioned artifact."""
    from fraud_ml.data.loader import read_frame, sha256_file
    from fraud_ml.training.experiment import start_run
    from fraud_ml.training.train import train_model

    settings = _settings(ctx)
    try:
        frame = read_frame(input_path)
        config = _training_config(settings, seed, model_version, fast, permutation)
        with start_run(
            enabled=settings.mlflow_enabled and not no_mlflow,
            tracking_uri=settings.mlflow_tracking_uri,
            experiment=settings.mlflow_experiment,
            run_name=f"{model_version}-{seed}",
        ) as tracker:
            result = train_model(
                frame,
                config,
                model_root=output or settings.model_dir,
                reports_dir=reports or settings.reports_dir,
                data_name=input_path.name,
                data_version=sha256_file(input_path),
                tracker=tracker,
            )
    except FraudMLError as exc:
        _fail(exc)
        return
    _print_training(result)


@app.command()
def evaluate(
    ctx: typer.Context,
    model: Annotated[Path, typer.Option("--model", "-m")] = Path("models/latest"),
    input_path: Annotated[
        Path, typer.Option("--input", "-i", help="Labelled dataset")
    ] = DEFAULT_DATA,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("reports/evaluation"),
) -> None:
    """Evaluate a saved artifact on a labelled dataset at its stored threshold."""
    from fraud_ml.contracts.transaction import LABEL_COLUMNS
    from fraud_ml.data.loader import read_frame
    from fraud_ml.data.validation import validate_frame
    from fraud_ml.scoring.scorer import Scorer
    from fraud_ml.training.evaluation import compute_metrics, plot_confusion_matrix, segment_metrics

    settings = _settings(ctx)
    try:
        scorer = Scorer.from_path(model)
        frame = read_frame(input_path)
        validate_frame(frame, require_labels=True, strict=True)
        x = frame.drop(columns=list(LABEL_COLUMNS))
        y = frame["is_fraud"].astype(bool).to_numpy()
        p = scorer.probabilities(x)
    except FraudMLError as exc:
        _fail(exc)
        return
    metrics = compute_metrics(y, p, scorer.policy.threshold, settings.cost_model)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    segment_metrics(x, y, p, scorer.policy.threshold, settings.cost_model).to_csv(
        output / "segment_metrics.csv", index=False
    )
    plot_confusion_matrix(
        metrics, output / "confusion_matrix.png", f"{scorer.model_version} on {input_path.name}"
    )
    console.print_json(metrics.model_dump_json())
    console.print(f"[green]wrote[/green] {output}/")


@app.command("score-file")
def score_file_cmd(
    model: Annotated[Path, typer.Option("--model", "-m")] = Path("models/latest"),
    input_path: Annotated[Path, typer.Option("--input", "-i")] = Path(
        "data/samples/scoring.parquet"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("reports/scores.parquet"),
) -> None:
    """Batch-score a Parquet / JSON Lines / CSV file."""
    from fraud_ml.scoring.batch import score_file
    from fraud_ml.scoring.scorer import Scorer

    try:
        summary = score_file(Scorer.from_path(model), input_path, output)
    except FraudMLError as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(summary.to_dict()))


@app.command()
def score(
    model: Annotated[Path, typer.Option("--model", "-m")] = Path("models/latest"),
    input_path: Annotated[
        Path, typer.Option("--input", "-i", help="JSON file with one transaction")
    ] = Path("examples/transaction.json"),
) -> None:
    """Score a single transaction from a JSON file."""
    from fraud_ml.contracts.transaction import TransactionFeatures
    from fraud_ml.scoring.scorer import Scorer

    try:
        transaction = TransactionFeatures.model_validate_json(
            input_path.read_text(encoding="utf-8")
        )
        response = Scorer.from_path(model).score_one(transaction)
    except FraudMLError as exc:
        _fail(exc)
        return
    except ValueError as exc:
        err_console.print(f"[red]error[/red] INVALID_INPUT: {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(response.model_dump_json())


@app.command("drift-check")
def drift_check_cmd(
    reference: Annotated[Path, typer.Option("--reference", "-r")] = DEFAULT_DATA,
    current: Annotated[Path, typer.Option("--current", "-c")] = Path(
        "data/samples/scoring.parquet"
    ),
    model: Annotated[
        Path | None, typer.Option("--model", "-m", help="Also compare score distributions")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the JSON report here")
    ] = None,
) -> None:
    """Compare feature (and optionally score) distributions between two datasets."""
    from fraud_ml.contracts.transaction import LABEL_COLUMNS
    from fraud_ml.data.loader import read_frame
    from fraud_ml.monitoring.drift import drift_check
    from fraud_ml.scoring.scorer import Scorer

    try:
        ref = read_frame(reference)
        cur = read_frame(current)
        score_column = None
        if model is not None:
            scorer = Scorer.from_path(model)
            ref = ref.assign(
                risk_probability=scorer.probabilities(
                    ref.drop(columns=[c for c in LABEL_COLUMNS if c in ref.columns])
                )
            )
            cur = cur.assign(
                risk_probability=scorer.probabilities(
                    cur.drop(columns=[c for c in LABEL_COLUMNS if c in cur.columns])
                )
            )
            score_column = "risk_probability"
        report = drift_check(ref, cur, score_column=score_column)
    except FraudMLError as exc:
        _fail(exc)
        return
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print_json(json.dumps(report.summary()))


@app.command("model-info")
def model_info(
    model: Annotated[Path, typer.Option("--model", "-m")] = Path("models/latest"),
) -> None:
    """Show the metadata of a saved artifact."""
    from fraud_ml.models.registry import read_metadata

    try:
        metadata = read_metadata(model)
    except FraudMLError as exc:
        _fail(exc)
        return
    console.print_json(metadata.info().model_dump_json())


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
) -> None:
    """Run the scoring API with uvicorn."""
    import uvicorn

    uvicorn.run("fraud_ml.main:app", host=host, port=port, log_level="info")


@app.command()
def demo(
    ctx: typer.Context,
    transactions: Annotated[int, typer.Option("--transactions", "-n", min=100)] = 50_000,
    fraud_rate: Annotated[float, typer.Option("--fraud-rate")] = 0.01,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    workdir: Annotated[
        Path, typer.Option("--workdir", help="Where data, models and reports go")
    ] = Path("."),
    fast: Annotated[bool, typer.Option("--fast")] = False,
    no_mlflow: Annotated[bool, typer.Option("--no-mlflow")] = False,
) -> None:
    """Generate -> train -> evaluate -> score (single + batch) -> drift check."""
    from fraud_ml.contracts.transaction import LABEL_COLUMNS, TransactionFeatures
    from fraud_ml.data.loader import read_frame, sha256_file, write_frame
    from fraud_ml.data.synthetic import GeneratorConfig, generate_dataset
    from fraud_ml.monitoring.drift import drift_check
    from fraud_ml.scoring.batch import score_file
    from fraud_ml.scoring.scorer import Scorer
    from fraud_ml.training.experiment import start_run
    from fraud_ml.training.train import train_model

    settings = _settings(ctx)
    data_dir = workdir / "data" / "generated"
    model_root = workdir / "models"
    reports_dir = workdir / "reports"
    try:
        console.rule("[bold]1. synthetic data")
        frame, summary = generate_dataset(
            GeneratorConfig(transactions=transactions, fraud_rate=fraud_rate, seed=seed)
        )
        data_path = write_frame(frame, data_dir / "transactions.parquet")
        console.print(
            f"Dataset: {summary.transactions:,} transactions, fraud rate {summary.fraud_rate:.2%}, {summary.accounts:,} accounts, {summary.start[:10]} to {summary.end[:10]}"
        )
        console.print(f"Fraud types: {summary.fraud_type_counts}")

        console.rule("[bold]2. temporal split, training, calibration, threshold, evaluation")
        config = _training_config(settings, seed, "fraud-risk/1.0.0", fast, permutation=False)
        with start_run(
            enabled=settings.mlflow_enabled and not no_mlflow,
            tracking_uri=settings.mlflow_tracking_uri
            if workdir == Path(".")
            else f"file:{(workdir / 'mlruns').as_posix()}",
            experiment=settings.mlflow_experiment,
            run_name=f"demo-{seed}",
        ) as tracker:
            result = train_model(
                frame,
                config,
                model_root=model_root,
                reports_dir=reports_dir,
                data_name=data_path.name,
                data_version=sha256_file(data_path),
                tracker=tracker,
            )
        for s in result.split_summary.values():
            console.print(
                f"  {s['name']:<10} {s['rows']:>7,} rows  {s['start'][:10]} to {s['end'][:10]}  positives {s['positives']} ({s['positive_rate']:.2%})"
            )
        _print_training(result)

        console.rule("[bold]3. score one transaction")
        scorer = Scorer.from_path(model_root / "latest")
        test_rows = frame.sort_values("transaction_timestamp").tail(int(len(frame) * 0.15))
        example = (
            test_rows[test_rows["is_fraud"]].iloc[-1]
            if test_rows["is_fraud"].any()
            else test_rows.iloc[-1]
        )
        payload = {
            k: v
            for k, v in example.drop(labels=list(LABEL_COLUMNS)).to_dict().items()
            if not (isinstance(v, float) and math.isnan(v))
        }
        payload["transaction_timestamp"] = example["transaction_timestamp"].isoformat()
        response = scorer.score_one(TransactionFeatures.model_validate(payload))
        console.print_json(response.model_dump_json())

        console.rule("[bold]4. batch scoring (latest 15% without labels)")
        scoring_path = write_frame(
            test_rows.drop(columns=list(LABEL_COLUMNS)), data_dir / "scoring.parquet"
        )
        batch = score_file(scorer, scoring_path, reports_dir / "scores.parquet")
        console.print_json(json.dumps(batch.to_dict()))

        console.rule("[bold]5. drift check (training period vs latest period)")
        train_rows = (
            frame.sort_values("transaction_timestamp")
            .head(int(len(frame) * 0.70))
            .drop(columns=list(LABEL_COLUMNS))
        )
        current = read_frame(scoring_path)
        ref = train_rows.assign(risk_probability=scorer.probabilities(train_rows))
        cur = current.assign(risk_probability=scorer.probabilities(current))
        report = drift_check(ref, cur, score_column="risk_probability")
        (reports_dir / "drift_report.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        console.print_json(json.dumps(report.summary()))
    except FraudMLError as exc:
        _fail(exc)
        return
    console.rule("[bold green]demo complete")
    console.print(f"model: {model_root / 'latest'}  reports: {reports_dir}")


if __name__ == "__main__":  # pragma: no cover
    app()
