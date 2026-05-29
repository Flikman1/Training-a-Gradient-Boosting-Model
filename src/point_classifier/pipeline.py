from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from .data import generate_dataset, get_features_target
from .metrics import calculate_metrics, format_metrics_summary, save_metrics
from .model import save_model, train_model
from .visualization import (
    plot_dataset,
    plot_decision_boundary,
    plot_predictions,
    plot_process_overview,
    plot_train_test_split,
)


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for the synthetic point classification pipeline."""

    n_samples: int = 10_000
    radius: float = 1.0
    shift: float = 1.5
    test_size: float = 0.25
    random_state: int = 42
    model_output_path: Path = Path("artifacts/models/gradient_boosting_model.joblib")
    plots_dir: Path = Path("artifacts/plots")
    metrics_output_path: Path = Path("reports/metrics.json")


@dataclass(slots=True)
class PipelineResult:
    """Artifacts produced by a pipeline run."""

    metrics: dict[str, Any]
    summary: str
    model_path: Path
    plots_dir: Path
    metrics_path: Path
    process_overview_path: Path


def _validate_pipeline_config(config: PipelineConfig) -> None:
    if not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the full dataset generation, training, evaluation and save routine."""
    _validate_pipeline_config(config)

    dataset = generate_dataset(
        n_samples=config.n_samples,
        radius=config.radius,
        shift=config.shift,
        random_state=config.random_state,
    )
    features, target = get_features_target(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        random_state=config.random_state,
    )
    predictions = model.predict(X_test)

    metrics = calculate_metrics(y_true=y_test, y_pred=predictions)
    summary = format_metrics_summary(metrics)

    plot_dataset(dataset, config.plots_dir / "synthetic_dataset.png")
    plot_train_test_split(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        path=config.plots_dir / "train_test_split.png",
    )
    plot_predictions(
        X_test=X_test,
        y_true=y_test,
        y_pred=predictions,
        path=config.plots_dir / "test_predictions.png",
    )
    plot_decision_boundary(
        model=model,
        X_test=X_test,
        y_test=y_test,
        path=config.plots_dir / "decision_boundary.png",
    )
    process_overview_path = config.plots_dir / "process_overview.png"
    plot_process_overview(
        dataset=dataset,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=predictions,
        model=model,
        metrics=metrics,
        path=process_overview_path,
    )

    save_model(model, config.model_output_path)
    save_metrics(metrics, config.metrics_output_path)

    return PipelineResult(
        metrics=metrics,
        summary=summary,
        model_path=config.model_output_path,
        plots_dir=config.plots_dir,
        metrics_path=config.metrics_output_path,
        process_overview_path=process_overview_path,
    )
