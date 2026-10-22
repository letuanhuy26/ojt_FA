from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_CANDIDATE = "regime_dynamic_gate_lstm"
MODEL_BASELINE = "random_forest"


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block_length + 1, size=math.ceil(n / block_length))
    blocks = [np.arange(start, start + block_length) for start in starts]
    return np.concatenate(blocks)[:n]


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probability >= threshold).astype(np.int8)
    return {
        "mcc": float(matthews_corrcoef(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "accuracy": float(accuracy_score(y, predicted)),
        "auroc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def main() -> None:
    analysis = yaml.safe_load((ROOT / "configs/heldout_analysis.yaml").read_text(encoding="utf-8"))
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))
    threshold = float(analysis["threshold"])
    if threshold != float(protocol["evaluation"]["threshold"]):
        raise SystemExit("Held-out analysis threshold differs from the frozen protocol.")
    replicates = int(analysis["bootstrap"]["replicates"])
    block_length = int(analysis["bootstrap"]["block_length_sessions"])
    seed = int(analysis["bootstrap"]["seed"])
    confidence_level = float(analysis["bootstrap"]["confidence_level"])
    alpha = 1.0 - confidence_level
    if replicates <= 0 or block_length <= 0:
        raise SystemExit("replicates and block length must be positive")
    output_dir = ROOT / "results/heldout_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "analysis": "paired moving-block bootstrap, candidate minus fixed Random Forest",
        "replicates": replicates,
        "block_length_sessions": block_length,
        "seed": seed,
        "threshold": threshold,
        "confidence_level": confidence_level,
        "interval": "paired moving-block bootstrap percentile interval; no p-value computed",
        "periods": {},
    }
    detailed_rows = []

    for split in ("confirmation", "final"):
        run_path = ROOT / "results" / split / "run.json"
        pred_path = ROOT / "results" / split / "heldout_predictions.csv"
        if not run_path.exists() or not pred_path.exists():
            raise SystemExit(f"Missing completed {split} evaluation; refusing partial inference.")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("status") != "completed" or run.get("split") != split:
            raise SystemExit(f"Invalid or incomplete {split} run manifest.")
        frame = pd.read_csv(pred_path, parse_dates=["target_date"])
        candidate = frame.loc[frame["model"] == MODEL_CANDIDATE].sort_values("target_date")
        baseline = frame.loc[frame["model"] == MODEL_BASELINE].sort_values("target_date")
        candidate = candidate.reset_index(drop=True)
        baseline = baseline.reset_index(drop=True)
        if candidate.empty or len(candidate) != len(baseline):
            raise SystemExit(f"Candidate/RF prediction rows differ in {split}.")
        if not candidate["target_date"].equals(baseline["target_date"]):
            raise SystemExit(f"Candidate/RF forecast dates do not align in {split}.")
        if not np.array_equal(candidate["y_true"].to_numpy(), baseline["y_true"].to_numpy()):
            raise SystemExit(f"Candidate/RF target labels do not align in {split}.")

        y = candidate["y_true"].to_numpy(dtype=np.int8)
        p_candidate = candidate["probability"].to_numpy(dtype=np.float64)
        p_baseline = baseline["probability"].to_numpy(dtype=np.float64)
        if len(np.unique(y)) != 2:
            raise SystemExit(f"{split} period must contain both target classes for AUROC.")
        if not (np.isfinite(p_candidate).all() and np.isfinite(p_baseline).all()):
            raise SystemExit(f"Non-finite probabilities found in {split}.")
        if not ((0 <= p_candidate).all() and (p_candidate <= 1).all()
                and (0 <= p_baseline).all() and (p_baseline <= 1).all()):
            raise SystemExit(f"Probabilities outside [0, 1] found in {split}.")
        observed = {
            MODEL_CANDIDATE: metrics(y, p_candidate, threshold),
            MODEL_BASELINE: metrics(y, p_baseline, threshold),
        }
        differences = {
            key: observed[MODEL_CANDIDATE][key] - observed[MODEL_BASELINE][key]
            for key in observed[MODEL_CANDIDATE]
        }
        boot = {key: [] for key in ("mcc", "balanced_accuracy", "auroc", "brier", "log_loss")}
        rng = np.random.default_rng(seed)
        valid_replicates = 0
        for _ in range(replicates):
            indices = moving_block_indices(len(y), min(block_length, len(y)), rng)
            if len(np.unique(y[indices])) != 2:
                continue
            candidate_metrics = metrics(y[indices], p_candidate[indices], threshold)
            baseline_metrics = metrics(y[indices], p_baseline[indices], threshold)
            for key in boot:
                boot[key].append(candidate_metrics[key] - baseline_metrics[key])
            valid_replicates += 1
        if valid_replicates < math.ceil(0.95 * replicates):
            raise SystemExit(
                f"Only {valid_replicates}/{replicates} bootstrap replicates were valid in {split}; "
                "refusing unstable intervals."
            )

        intervals = {
            key: {
                "lower": float(np.quantile(values, alpha / 2)),
                "upper": float(np.quantile(values, 1 - alpha / 2)),
            }
            for key, values in boot.items()
        }
        output["periods"][split] = {
            "n": int(len(y)),
            "observed_positive_rate": float(y.mean()),
            "bootstrap_seed": seed,
            "valid_bootstrap_replicates": valid_replicates,
            "candidate_metrics": observed[MODEL_CANDIDATE],
            "random_forest_metrics": observed[MODEL_BASELINE],
            "candidate_minus_random_forest": differences,
            "paired_moving_block_bootstrap_ci": intervals,
        }
        for model, values in observed.items():
            detailed_rows.append({"split": split, "model": model, "n": len(y), **values})

    output_path = output_dir / "paired_bootstrap_summary.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    pd.DataFrame(detailed_rows).to_csv(output_dir / "point_metrics.csv", index=False, float_format="%.10g")
    print(f"Heldout_Analysis=PASS replicates={replicates} block_length={block_length}")
    for split, result in output["periods"].items():
        delta = result["candidate_minus_random_forest"]
        ci = result["paired_moving_block_bootstrap_ci"]["mcc"]
        print(f"{split}_candidate_mcc={result['candidate_metrics']['mcc']:.6f}")
        print(f"{split}_rf_mcc={result['random_forest_metrics']['mcc']:.6f}")
        print(f"{split}_delta_mcc={delta['mcc']:.6f} ci={confidence_level:.0%}=[{ci['lower']:.6f},{ci['upper']:.6f}]")


if __name__ == "__main__":
    main()
