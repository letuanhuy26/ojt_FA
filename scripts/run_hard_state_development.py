from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import walkforward
from src.evaluation.walkforward import evaluate_models, load_yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    analysis_path = ROOT / "configs/hard_state_analysis.yaml"
    analysis = load_yaml(analysis_path)
    config = load_yaml(ROOT / "configs/ours.yaml")
    protocol = load_yaml(ROOT / "configs/protocol.yaml")
    if analysis["split"] != "development":
        raise SystemExit("Hard-state control is restricted to development.")
    if analysis["model"] != config["model"]:
        raise SystemExit("Hard-state control must keep the frozen candidate architecture.")

    original_prepare_fold = walkforward.prepare_fold

    def prepare_hard_state_fold(*args, **kwargs):
        fold = original_prepare_fold(*args, **kwargs)
        for key in ("p_sequences_train", "p_sequences_val"):
            soft_posterior = fold[key]
            hard_index = soft_posterior.argmax(axis=-1)
            fold[key] = np.eye(soft_posterior.shape[-1], dtype=np.float32)[hard_index]
        return fold

    metric_runs = []
    walkforward.prepare_fold = prepare_hard_state_fold
    try:
        for seed_base in analysis["model_seed_bases"]:
            seed_config = dict(config)
            seed_config["seed"] = int(seed_base)
            metrics, _ = evaluate_models(
                [analysis["model"]], config=seed_config, protocol=protocol, split="development"
            )
            metrics["model_seed_base"] = int(seed_base)
            metric_runs.append(metrics)
    finally:
        walkforward.prepare_fold = original_prepare_fold

    hard = pd.concat(metric_runs, ignore_index=True)
    soft_path = ROOT / "results/development/gate_diagnostic_fold_summary.csv"
    soft = pd.read_csv(soft_path)
    soft = soft[["model_seed_base", "fold_year", "replayed_mcc"]].rename(
        columns={"replayed_mcc": "soft_posterior_mcc"}
    )
    hard = hard.rename(columns={"mcc": "hard_state_mcc"})
    paired = hard.merge(soft, on=["model_seed_base", "fold_year"], how="inner", validate="one_to_one")
    paired["hard_minus_soft_mcc"] = paired["hard_state_mcc"] - paired["soft_posterior_mcc"]
    if len(paired) != len(hard) or len(hard) != len(analysis["model_seed_bases"]) * len(protocol["evaluation"]["development_years"]):
        raise SystemExit("Hard-state and soft-posterior rows do not cover the same development folds/seeds.")

    fold_summary = paired.groupby("fold_year", sort=True).agg(
        seeds=("model_seed_base", "nunique"),
        hard_mean_mcc=("hard_state_mcc", "mean"),
        hard_sd_seed_mcc=("hard_state_mcc", "std"),
        soft_mean_mcc=("soft_posterior_mcc", "mean"),
        mean_hard_minus_soft_mcc=("hard_minus_soft_mcc", "mean"),
        positive_seed_deltas=("hard_minus_soft_mcc", lambda values: int((values > 0).sum())),
    ).reset_index()
    summary = pd.DataFrame([{
        "fold_seed_runs": len(paired),
        "seed_bases": len(analysis["model_seed_bases"]),
        "hard_mean_mcc": float(paired["hard_state_mcc"].mean()),
        "hard_sd_fold_seed_mcc": float(paired["hard_state_mcc"].std(ddof=1)),
        "soft_mean_mcc": float(paired["soft_posterior_mcc"].mean()),
        "soft_sd_fold_seed_mcc": float(paired["soft_posterior_mcc"].std(ddof=1)),
        "mean_hard_minus_soft_mcc": float(paired["hard_minus_soft_mcc"].mean()),
        "positive_hard_minus_soft_runs": int((paired["hard_minus_soft_mcc"] > 0).sum()),
        "negative_hard_minus_soft_runs": int((paired["hard_minus_soft_mcc"] < 0).sum()),
    }])
    out = ROOT / "results/development"
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "hard_state_fold_seed_metrics.csv", index=False, float_format="%.10g")
    fold_summary.to_csv(out / "hard_state_fold_summary.csv", index=False, float_format="%.10g")
    summary.to_csv(out / "hard_state_summary.csv", index=False, float_format="%.10g")
    manifest = {
        "status": analysis["analysis_status"],
        "split": "development",
        "control": analysis["control"],
        "model": analysis["model"],
        "model_seed_bases": analysis["model_seed_bases"],
        "candidate_changed": False,
        "selection_or_tuning_use": False,
        "heldout_evaluation": False,
        "matched_soft_control": "development gate diagnostics with the same fold and seed base",
        "hard_minus_soft_mean_mcc": float(paired["hard_minus_soft_mcc"].mean()),
        "positive_hard_minus_soft_runs": int((paired["hard_minus_soft_mcc"] > 0).sum()),
        "negative_hard_minus_soft_runs": int((paired["hard_minus_soft_mcc"] < 0).sum()),
        "config_sha256": sha256(ROOT / "configs/ours.yaml"),
        "protocol_sha256": sha256(ROOT / "configs/protocol.yaml"),
        "model_sha256": sha256(ROOT / "src/models/ours.py"),
        "evaluator_sha256": sha256(ROOT / "src/evaluation/walkforward.py"),
        "analysis_config_sha256": sha256(analysis_path),
        "files": {
            "fold_seed_metrics": "results/development/hard_state_fold_seed_metrics.csv",
            "fold_summary": "results/development/hard_state_fold_summary.csv",
            "summary": "results/development/hard_state_summary.csv",
        },
    }
    (out / "hard_state_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(fold_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Hard_State_Status={manifest['status']}")


if __name__ == "__main__":
    main()
