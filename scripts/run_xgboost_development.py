from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import xgboost
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.walkforward import (
    chronological_folds,
    load_examples,
    load_yaml,
    metric_row,
    prepare_fold,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    analysis_path = ROOT / "configs/xgboost_analysis.yaml"
    config = load_yaml(analysis_path)
    protocol = load_yaml(ROOT / "configs/protocol.yaml")
    ours = load_yaml(ROOT / "configs/ours.yaml")
    if config["split"] != "development":
        raise SystemExit("The post-hoc XGBoost run is restricted to development.")
    if config["decision_threshold"] != protocol["evaluation"]["threshold"]:
        raise SystemExit("XGBoost must use the protocol's fixed classification threshold.")

    frame = load_examples()
    fold_rows: list[dict] = []
    started_all = time.perf_counter()
    for year, start, end in chronological_folds(protocol, split="development"):
        fold = prepare_fold(frame, protocol, year, start, end, int(ours["window"]))
        seed = int(config["seed_base"]) + year
        model = xgboost.XGBClassifier(
            random_state=seed,
            **config["parameters"],
        )
        start_fold = time.perf_counter()
        model.fit(fold["x_train_rows"], fold["y_train_rows"])
        probability = model.predict_proba(fold["x_val_rows"])[:, 1]
        scores = metric_row(
            fold["y_val_rows"], probability, threshold=float(config["decision_threshold"])
        )
        fold_rows.append({
            "model": "xgboost",
            "fold_year": year,
            "seed": seed,
            "hmm_iterations": fold["hmm_iterations"],
            "hmm_loglik_per_obs": fold["hmm_train_log_likelihood_per_obs"],
            **scores,
            "runtime_seconds": time.perf_counter() - start_fold,
        })

    metrics = pd.DataFrame(fold_rows)
    if metrics["fold_year"].tolist() != list(protocol["evaluation"]["development_years"]):
        raise SystemExit("XGBoost development folds differ from the locked protocol.")
    summary = metrics.agg({
        "mcc": ["mean", "std", "min", "max"],
        "balanced_accuracy": ["mean", "std"],
        "auroc": ["mean", "std"],
        "brier": ["mean", "std"],
        "log_loss": ["mean", "std"],
    })
    out = ROOT / "results/development"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "xgboost_fold_metrics.csv", index=False, float_format="%.10g")
    summary.to_csv(out / "xgboost_summary.csv", float_format="%.10g")
    manifest = {
        "status": config["analysis_status"],
        "split": "development",
        "model": "xgboost",
        "package_version": xgboost.__version__,
        "candidate_changed": False,
        "selection_or_tuning_use": False,
        "heldout_evaluation": False,
        "seed_base": int(config["seed_base"]),
        "parameters": config["parameters"],
        "threshold": float(config["decision_threshold"]),
        "fold_mcc": {str(int(row.fold_year)): float(row.mcc) for row in metrics.itertuples()},
        "mean_development_mcc": float(metrics["mcc"].mean()),
        "sd_fold_mcc": float(metrics["mcc"].std(ddof=1)),
        "mean_balanced_accuracy": float(metrics["balanced_accuracy"].mean()),
        "mean_auroc": float(metrics["auroc"].mean()),
        "elapsed_seconds": time.perf_counter() - started_all,
        "config_sha256": sha256(analysis_path),
        "protocol_sha256": sha256(ROOT / "configs/protocol.yaml"),
        "evaluator_sha256": sha256(ROOT / "src/evaluation/walkforward.py"),
        "files": {
            "fold_metrics": "results/development/xgboost_fold_metrics.csv",
            "summary": "results/development/xgboost_summary.csv",
        },
    }
    (out / "xgboost_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics[["fold_year", "n", "mcc", "balanced_accuracy", "auroc", "runtime_seconds"]].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    print(f"XGBoost_Mean_MCC={manifest['mean_development_mcc']:.8f}")
    print(f"XGBoost_Version={manifest['package_version']}")
    print(f"XGBoost_Status={manifest['status']}")


if __name__ == "__main__":
    main()
