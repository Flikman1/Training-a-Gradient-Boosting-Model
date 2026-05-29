from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    """Train a GradientBoostingClassifier on the provided data."""
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def save_model(model: GradientBoostingClassifier, path: Path) -> None:
    """Persist a trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> GradientBoostingClassifier:
    """Load a trained model from disk."""
    loaded_model = joblib.load(path)
    return loaded_model
