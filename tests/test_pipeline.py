from __future__ import annotations

import json

from point_classifier.pipeline import PipelineConfig, run_pipeline


def test_pipeline_creates_expected_artifacts(tmp_path) -> None:
    model_path = tmp_path / "models" / "gradient_boosting_model.joblib"
    plots_dir = tmp_path / "plots"
    metrics_path = tmp_path / "reports" / "metrics.json"

    config = PipelineConfig(
        n_samples=100,
        radius=1.0,
        shift=1.5,
        test_size=0.25,
        random_state=42,
        model_output_path=model_path,
        plots_dir=plots_dir,
        metrics_output_path=metrics_path,
    )

    result = run_pipeline(config)

    assert result.model_path.exists()
    assert result.metrics_path.exists()
    assert any(result.plots_dir.glob("*.png"))
    assert result.process_overview_path.exists()

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert {"accuracy", "precision", "recall", "f1_score"}.issubset(metrics)
