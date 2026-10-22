from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.walkforward import evaluate_models


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    analysis_path = ROOT / "configs/ablation_analysis.yaml"
    base_path = ROOT / "configs/ours.yaml"
    protocol_path = ROOT / "configs/protocol.yaml"
    analysis = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if analysis["split"] != "development":
        raise SystemExit("Gate ablations are restricted to development.")
    if base["model"] != "regime_dynamic_gate_lstm":
        raise SystemExit("Base configuration no longer identifies the frozen RC-SFG candidate.")

    metrics_rows = []
    started = time.perf_counter()
    reference = None
    for name, overrides in analysis["variants"].items():
        config = dict(base)
        config.update(overrides)
        invariant_fields = [key for key in base if key not in {"lambda_sparsity", "lambda_stability"}]
        if any(config[key] != base[key] for key in invariant_fields):
            raise SystemExit(f"Unexpected config change in ablation {name}.")
        if name == "selected_candidate":
            if overrides["lambda_sparsity"] != base["lambda_sparsity"] or overrides["lambda_stability"] != base["lambda_stability"]:
                raise SystemExit("Selected-candidate reference does not match the frozen config.")
        model_metrics, _ = evaluate_models([base["model"]], config=config, protocol=protocol, split="development")
        if len(model_metrics) != len(protocol["evaluation"]["development_years"]):
            raise SystemExit(f"Ablation {name} did not return one result per development fold.")
        if not model_metrics["mcc"].notna().all():
            raise SystemExit(f"Ablation {name} returned a missing primary metric.")
        if name == "selected_candidate":
            reference = float(model_metrics["mcc"].mean())
            if abs(reference - 0.0331907033) >= 1e-8:
                raise SystemExit(f"Selected-candidate replay changed: observed mean MCC={reference:.10f}.")
        for _, row in model_metrics.iterrows():
            metrics_rows.append({
                "variant": name,
                "lambda_sparsity": overrides["lambda_sparsity"],
                "lambda_stability": overrides["lambda_stability"],
                **row.to_dict(),
            })

    out = ROOT / "results/development"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(metrics_rows)
    frame.to_csv(out / "gate_ablation_fold_metrics.csv", index=False, float_format="%.10g")
    summary = frame.groupby(["variant", "lambda_sparsity", "lambda_stability"], sort=False).agg(
        folds=("fold_year", "count"),
        mean_mcc=("mcc", "mean"),
        sd_fold_mcc=("mcc", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_auroc=("auroc", "mean"),
    ).reset_index()
    summary.to_csv(out / "gate_ablation_summary.csv", index=False, float_format="%.10g")
    manifest = {
        "status": analysis["analysis_status"],
        "split": "development",
        "candidate_changed": False,
        "selection_or_tuning_use": False,
        "primary_metric": "mean MCC across the five locked development folds",
        "variant_count": int(summary.shape[0]),
        "config_sha256": sha256(base_path),
        "analysis_config_sha256": sha256(analysis_path),
        "evaluator_sha256": sha256(ROOT / "src/evaluation/walkforward.py"),
        "elapsed_seconds": time.perf_counter() - started,
        "summary_file": "results/development/gate_ablation_summary.csv",
        "fold_file": "results/development/gate_ablation_fold_metrics.csv",
    }
    (out / "gate_ablation_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Ablation_Status={manifest['status']}")
    print(f"Ablation_Elapsed_Seconds={manifest['elapsed_seconds']:.3f}")


if __name__ == "__main__":
    main()
