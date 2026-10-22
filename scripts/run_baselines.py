from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.walkforward import evaluate_models, summarize_metrics, write_run_metadata


DEFAULT_MODELS = [
    "majority",
    "previous_direction",
    "logistic",
    "random_forest",
    "ungated_lstm",
    "static_gate_lstm",
    "dynamic_gate_lstm",
    "regime_as_feature_lstm",
    "regime_dynamic_gate_lstm",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predeclared baselines on development folds only.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated model names")
    args = parser.parse_args()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    metrics, predictions = evaluate_models(models)
    out = ROOT / "results/development"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "baseline_fold_metrics.csv", index=False, float_format="%.10g")
    predictions.to_csv(out / "baseline_predictions.csv", index=False, float_format="%.10g")
    summary = summarize_metrics(metrics)
    summary.to_csv(out / "baseline_summary.csv", index=False, float_format="%.10g")
    write_run_metadata(out / "baseline_run.json", metrics, models)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved development-only outputs in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
