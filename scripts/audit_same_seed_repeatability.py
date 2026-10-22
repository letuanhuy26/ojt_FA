from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import FEATURE_COLUMNS
from src.evaluation.walkforward import evaluate_models, load_examples, load_yaml


def main() -> None:
    frozen_path = ROOT / "data/processed/SPY_features_1993-2025.csv"
    transfer_path = ROOT / "data/processed/market_transfer_20261004/SPY_features.csv"
    frozen = pd.read_csv(frozen_path, parse_dates=["date", "target_date"])
    rebuilt = pd.read_csv(transfer_path, parse_dates=["date", "target_date"])
    if len(frozen) != len(rebuilt):
        raise SystemExit(f"Feature row count mismatch: {len(frozen)} vs {len(rebuilt)}")
    date_parity = frozen["date"].equals(rebuilt["date"]) and frozen["target_date"].equals(rebuilt["target_date"])
    target_class_parity = np.array_equal(
        frozen["target"].to_numpy(dtype=np.int8), rebuilt["target"].to_numpy(dtype=np.int8)
    )
    target_return_delta = float(np.max(np.abs(
        frozen["target_return"].to_numpy(dtype=float) - rebuilt["target_return"].to_numpy(dtype=float)
    )))
    maximum_feature_delta = max(
        float(np.max(np.abs(frozen[column].to_numpy(dtype=float) - rebuilt[column].to_numpy(dtype=float))))
        for column in FEATURE_COLUMNS
    )
    print(f"feature_rows={len(frozen)}")
    print(f"date_parity={date_parity}")
    print(f"target_class_parity={target_class_parity}")
    print(f"maximum_target_return_absolute_delta={target_return_delta:.12g}")
    print(f"maximum_feature_absolute_delta={maximum_feature_delta:.12g}")

    config = load_yaml(ROOT / "configs/ours.yaml")
    protocol = load_yaml(ROOT / "configs/protocol.yaml")
    historical = pd.read_csv(ROOT / "autoresearch/model-20261004-01/final-best-fold-metrics.csv")
    if "model" in historical.columns:
        historical = historical[historical["model"] == "regime_dynamic_gate_lstm"]
    historical = historical.sort_values("fold_year")["mcc"].to_numpy(dtype=float)
    metrics_a, predictions_a = evaluate_models([config["model"]], config=config, protocol=protocol)
    metrics_b, predictions_b = evaluate_models([config["model"]], config=config, protocol=protocol)
    metrics_a = metrics_a.sort_values("fold_year").reset_index(drop=True)
    metrics_b = metrics_b.sort_values("fold_year").reset_index(drop=True)
    predictions_a = predictions_a.sort_values("target_date").reset_index(drop=True)
    predictions_b = predictions_b.sort_values("target_date").reset_index(drop=True)
    probability_delta = float(np.max(np.abs(
        predictions_a["probability"].to_numpy(dtype=float)
        - predictions_b["probability"].to_numpy(dtype=float)
    )))
    score_a = metrics_a["mcc"].to_numpy(dtype=float)
    score_b = metrics_b["mcc"].to_numpy(dtype=float)
    print("historical_candidate_fold_mcc=" + ",".join(f"{value:.10f}" for value in historical))
    print("repeat_a_fold_mcc=" + ",".join(f"{value:.10f}" for value in score_a))
    print("repeat_b_fold_mcc=" + ",".join(f"{value:.10f}" for value in score_b))
    print(f"repeat_a_b_max_probability_delta={probability_delta:.12g}")
    print(f"repeat_a_b_metrics_equal={np.array_equal(score_a, score_b)}")
    print(f"repeat_a_historical_metrics_equal={np.array_equal(score_a, historical)}")
    print(f"device={sorted(metrics_a['device'].astype(str).unique().tolist())}")

    metrics_rebuilt, predictions_rebuilt = evaluate_models(
        [config["model"]], config=config, protocol=protocol, data_path=transfer_path
    )
    predictions_rebuilt = predictions_rebuilt.sort_values("target_date").reset_index(drop=True)
    original_probability = predictions_a["probability"].to_numpy(dtype=float)
    rebuilt_probability = predictions_rebuilt["probability"].to_numpy(dtype=float)
    print(f"original_vs_rebuilt_max_probability_delta={np.max(np.abs(original_probability - rebuilt_probability)):.12g}")
    print(f"original_vs_rebuilt_changed_threshold_count={int(np.count_nonzero((original_probability >= 0.5) != (rebuilt_probability >= 0.5)))}")
    print("rebuilt_fold_mcc=" + ",".join(
        f"{value:.10f}" for value in metrics_rebuilt.sort_values("fold_year")["mcc"].to_numpy(dtype=float)
    ))

    transfer_predictions = pd.read_csv(ROOT / "results/market_transfer_20261004/predictions.csv")
    transfer_predictions = transfer_predictions[
        (transfer_predictions["symbol"] == "SPY")
        & (transfer_predictions["split"] == "development")
        & (transfer_predictions["model"] == config["model"])
    ].sort_values("target_date").reset_index(drop=True)
    transfer_probability = transfer_predictions["probability"].to_numpy(dtype=float)
    print(f"rebuilt_vs_market_run_max_probability_delta={np.max(np.abs(rebuilt_probability - transfer_probability)):.12g}")
    print(f"rebuilt_vs_market_run_changed_threshold_count={int(np.count_nonzero((rebuilt_probability >= 0.5) != (transfer_probability >= 0.5)))}")


if __name__ == "__main__":
    main()
