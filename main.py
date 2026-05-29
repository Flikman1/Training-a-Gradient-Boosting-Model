from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from point_classifier.pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train a GradientBoostingClassifier on synthetic 2D point data."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10_000,
        help="Number of points to generate per class.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Radius of each circular point cloud.",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=1.5,
        help="Shift applied to the second class on both axes.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of samples reserved for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for data generation and model training.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/models/gradient_boosting_model.joblib"),
        help="Path where the trained model will be saved.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("artifacts/plots"),
        help="Directory where PNG plots will be saved.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("reports/metrics.json"),
        help="Path where metrics.json will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end training pipeline from the CLI."""
    args = parse_args()
    config = PipelineConfig(
        n_samples=args.n_samples,
        radius=args.radius,
        shift=args.shift,
        test_size=args.test_size,
        random_state=args.random_state,
        model_output_path=args.model_path,
        plots_dir=args.plots_dir,
        metrics_output_path=args.metrics_path,
    )
    result = run_pipeline(config)
    print(result.summary)
    print(f"Model saved to: {result.model_path}")
    print(f"Metrics saved to: {result.metrics_path}")
    print(f"Plots saved to: {result.plots_dir}")
    print(f"Process overview saved to: {result.process_overview_path}")


if __name__ == "__main__":
    main()
