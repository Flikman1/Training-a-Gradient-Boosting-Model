from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _create_prediction_grid(
    X_frame: pd.DataFrame,
    step: float = 0.02,
    padding: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create a rectangular feature grid for decision-surface visualizations."""
    x_min = X_frame["x"].min() - padding
    x_max = X_frame["x"].max() + padding
    y_min = X_frame["y"].min() - padding
    y_max = X_frame["y"].max() + padding

    x_grid, y_grid = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step),
    )
    grid_points = pd.DataFrame(
        {
            "x": x_grid.ravel(),
            "y": y_grid.ravel(),
        }
    )
    return x_grid, y_grid, grid_points


def plot_dataset(dataset: pd.DataFrame, path: Path) -> None:
    """Save a scatter plot of the full synthetic dataset."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        dataset["x"],
        dataset["y"],
        c=dataset["target"],
        cmap="viridis",
        alpha=0.6,
        s=12,
    )
    ax.set_title("Synthetic dataset")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(*scatter.legend_elements(), title="Class", loc="best")
    _save_figure(fig, path)


def plot_train_test_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    path: Path,
) -> None:
    """Save a plot that shows train and test samples."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        X_train["x"],
        X_train["y"],
        c=y_train,
        cmap="viridis",
        alpha=0.25,
        s=10,
        label="Train",
    )
    ax.scatter(
        X_test["x"],
        X_test["y"],
        c=y_test,
        cmap="viridis",
        alpha=0.9,
        s=20,
        marker="x",
        label="Test",
    )
    ax.set_title("Train/test split")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    _save_figure(fig, path)


def plot_predictions(
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    path: Path,
) -> None:
    """Save a plot of predicted classes on the test split."""
    correctness = y_true.to_numpy() == y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        X_test.loc[correctness, "x"],
        X_test.loc[correctness, "y"],
        c=y_pred[correctness],
        cmap="viridis",
        alpha=0.8,
        s=20,
        label="Correct prediction",
    )
    if (~correctness).any():
        ax.scatter(
            X_test.loc[~correctness, "x"],
            X_test.loc[~correctness, "y"],
            c=y_pred[~correctness],
            cmap="viridis",
            alpha=0.9,
            s=36,
            marker="x",
            label="Misclassified",
        )
    ax.set_title("Predicted classes on test data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    _save_figure(fig, path)


def plot_decision_boundary(
    model: GradientBoostingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    path: Path,
    step: float = 0.02,
    padding: float = 0.5,
) -> None:
    """Save a decision boundary plot for the trained classifier."""
    x_grid, y_grid, grid_points = _create_prediction_grid(
        X_frame=X_test,
        step=step,
        padding=padding,
    )
    grid_predictions = model.predict(grid_points).reshape(x_grid.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(x_grid, y_grid, grid_predictions, alpha=0.25, cmap="viridis")
    scatter = ax.scatter(
        X_test["x"],
        X_test["y"],
        c=y_test,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.3,
        s=18,
    )
    ax.set_title("Decision boundary")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(*scatter.legend_elements(), title="True class", loc="best")
    _save_figure(fig, path)


def plot_process_overview(
    dataset: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
    model: GradientBoostingClassifier,
    metrics: dict[str, object],
    path: Path,
) -> None:
    """Save a single figure that visualizes the full ML workflow."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    ax_dataset, ax_split, ax_predictions, ax_boundary, ax_confusion, ax_summary = axes.ravel()

    dataset_scatter = ax_dataset.scatter(
        dataset["x"],
        dataset["y"],
        c=dataset["target"],
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    ax_dataset.set_title("1. Synthetic dataset")
    ax_dataset.set_xlabel("x")
    ax_dataset.set_ylabel("y")
    ax_dataset.legend(*dataset_scatter.legend_elements(), title="Class", loc="best")

    ax_split.scatter(
        X_train["x"],
        X_train["y"],
        c=y_train,
        cmap="viridis",
        alpha=0.2,
        s=10,
        label="Train",
    )
    ax_split.scatter(
        X_test["x"],
        X_test["y"],
        c=y_test,
        cmap="viridis",
        alpha=0.85,
        s=24,
        marker="x",
        label="Test",
    )
    ax_split.set_title("2. Train/test split")
    ax_split.set_xlabel("x")
    ax_split.set_ylabel("y")
    ax_split.legend(loc="best")

    correctness = y_test.to_numpy() == y_pred
    ax_predictions.scatter(
        X_test.loc[correctness, "x"],
        X_test.loc[correctness, "y"],
        c=y_pred[correctness],
        cmap="viridis",
        alpha=0.8,
        s=24,
        label="Correct",
    )
    if (~correctness).any():
        ax_predictions.scatter(
            X_test.loc[~correctness, "x"],
            X_test.loc[~correctness, "y"],
            c=y_pred[~correctness],
            cmap="viridis",
            alpha=0.95,
            s=40,
            marker="x",
            label="Misclassified",
        )
    ax_predictions.set_title("3. Predictions on test set")
    ax_predictions.set_xlabel("x")
    ax_predictions.set_ylabel("y")
    ax_predictions.legend(loc="best")

    x_grid, y_grid, grid_points = _create_prediction_grid(X_frame=dataset[["x", "y"]])
    grid_predictions = model.predict(grid_points).reshape(x_grid.shape)
    ax_boundary.contourf(x_grid, y_grid, grid_predictions, alpha=0.25, cmap="viridis")
    boundary_scatter = ax_boundary.scatter(
        X_test["x"],
        X_test["y"],
        c=y_test,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.3,
        s=18,
    )
    ax_boundary.set_title("4. Decision boundary")
    ax_boundary.set_xlabel("x")
    ax_boundary.set_ylabel("y")
    ax_boundary.legend(
        *boundary_scatter.legend_elements(),
        title="True class",
        loc="best",
    )

    confusion = np.asarray(metrics["confusion_matrix"])
    confusion_image = ax_confusion.imshow(confusion, cmap="Blues")
    ax_confusion.set_title("5. Confusion matrix")
    ax_confusion.set_xlabel("Predicted label")
    ax_confusion.set_ylabel("True label")
    ax_confusion.set_xticks([0, 1], labels=["0", "1"])
    ax_confusion.set_yticks([0, 1], labels=["0", "1"])
    for row_index in range(confusion.shape[0]):
        for column_index in range(confusion.shape[1]):
            ax_confusion.text(
                column_index,
                row_index,
                f"{confusion[row_index, column_index]}",
                ha="center",
                va="center",
                color="black",
            )
    fig.colorbar(confusion_image, ax=ax_confusion, fraction=0.046, pad=0.04)

    ax_summary.axis("off")
    summary_lines = [
        "6. Pipeline summary",
        "",
        "Steps:",
        "1. Generate two synthetic point clouds",
        "2. Split into train and test subsets",
        "3. Train GradientBoostingClassifier",
        "4. Predict labels for the test set",
        "5. Evaluate the classifier",
        "",
        "Metrics:",
        f"Accuracy:  {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall:    {metrics['recall']:.4f}",
        f"F1-score:  {metrics['f1_score']:.4f}",
        "",
        f"Samples per class: {len(dataset) // 2}",
        f"Train size: {len(X_train)}",
        f"Test size: {len(X_test)}",
    ]
    ax_summary.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("Synthetic 2D Point Classification Workflow", fontsize=16, y=1.02)
    _save_figure(fig, path)
