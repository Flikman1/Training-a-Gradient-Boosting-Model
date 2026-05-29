from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

FEATURE_COLUMNS: Final[list[str]] = ["x", "y"]
TARGET_COLUMN: Final[str] = "target"


def _validate_generation_params(n_samples: int, radius: float) -> None:
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if radius <= 0:
        raise ValueError("radius must be greater than zero.")


def _sample_circle_points(
    n_samples: int,
    radius: float,
    center: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample 2D points uniformly inside a circle."""
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    radii = radius * np.sqrt(rng.random(n_samples))
    x_values = center[0] + radii * np.cos(angles)
    y_values = center[1] + radii * np.sin(angles)
    return np.column_stack((x_values, y_values))


def generate_dataset(
    n_samples: int = 10_000,
    radius: float = 1.0,
    shift: float = 1.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a shuffled binary classification dataset of 2D points.

    Parameters
    ----------
    n_samples:
        Number of samples generated for each class.
    radius:
        Radius of each circular point cloud.
    shift:
        Offset applied to the second class on both axes.
    random_state:
        Seed for deterministic data generation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ``x``, ``y`` and ``target``.
    """
    _validate_generation_params(n_samples=n_samples, radius=radius)

    rng = np.random.default_rng(random_state)
    class_zero_points = _sample_circle_points(
        n_samples=n_samples,
        radius=radius,
        center=(0.0, 0.0),
        rng=rng,
    )
    class_one_points = _sample_circle_points(
        n_samples=n_samples,
        radius=radius,
        center=(shift, shift),
        rng=rng,
    )

    features = np.vstack((class_zero_points, class_one_points))
    targets = np.concatenate(
        (
            np.zeros(n_samples, dtype=int),
            np.ones(n_samples, dtype=int),
        )
    )

    permutation = rng.permutation(features.shape[0])
    shuffled_features = features[permutation]
    shuffled_targets = targets[permutation]

    dataset = pd.DataFrame(shuffled_features, columns=FEATURE_COLUMNS)
    dataset[TARGET_COLUMN] = shuffled_targets
    return dataset


def get_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a generated dataset into features and target."""
    return dataset[FEATURE_COLUMNS].copy(), dataset[TARGET_COLUMN].copy()
