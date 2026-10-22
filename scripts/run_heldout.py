from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.evaluation.walkforward import evaluate_models, summarize_metrics


FREEZE_PATH = ROOT / "autoresearch/heldout-20261004-01/freeze.json"
MODELS = ["regime_dynamic_gate_lstm", "random_forest"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"HELDOUT_FAIL: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen candidate and fixed RF once on a held-out period.")
    parser.add_argument("--split", choices=["confirmation", "final"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="show the locked period and frozen files without evaluating")
    args = parser.parse_args()
    require(FREEZE_PATH.exists(), "candidate freeze manifest is missing")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    require(freeze.get("status") == "candidate_frozen_before_confirmation", "candidate freeze status is invalid")
    for name, expected_hash in freeze["file_sha256"].items():
        path = ROOT / name
        require(path.exists() and sha256(path) == expected_hash, f"frozen input changed: {name}")

    if args.split == "confirmation":
        confirmation_dir = ROOT / "results/confirmation"
        final_dir = ROOT / "results/final"
        require(not confirmation_dir.exists(), "confirmation has already been run; refusing a second look")
        require(not final_dir.exists(), "final results exist before confirmation")
    else:
        confirmation_dir = ROOT / "results/confirmation"
        confirmation_run = confirmation_dir / "run.json"
        require(confirmation_run.exists(), "final test requires a completed confirmation run")
        confirmation = json.loads(confirmation_run.read_text(encoding="utf-8"))
        require(confirmation.get("status") == "completed" and confirmation.get("split") == "confirmation",
                "confirmation run is incomplete or mismatched")
        final_dir = ROOT / "results/final"
        require(not final_dir.exists(), "final test has already been run; refusing a second look")

    out_dir = ROOT / "results" / args.split
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))
    evaluation = protocol["evaluation"]
    if args.split == "confirmation":
        expected_start = pd.Timestamp(evaluation["confirmation_start"])
        expected_end = pd.Timestamp(evaluation["confirmation_end_exclusive"])
    else:
        expected_start = pd.Timestamp(evaluation["final_start"])
        expected_end = pd.Timestamp(evaluation["final_end_exclusive"])
    if args.dry_run:
        print(f"Heldout_Dry_Run=PASS split={args.split} start={expected_start.date()} end_exclusive={expected_end.date()}")
        print(f"Models={','.join(MODELS)} output={out_dir.relative_to(ROOT)}")
        print(f"Frozen_config_sha256={freeze['file_sha256']['configs/ours.yaml']}")
        return

    out_dir.mkdir(parents=False, exist_ok=False)
    started = {
        "status": "running",
        "split": args.split,
        "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "frozen_commit": freeze["frozen_commit"],
        "config_sha256": freeze["file_sha256"]["configs/ours.yaml"],
    }
    (out_dir / "started.json").write_text(json.dumps(started, indent=2), encoding="utf-8")

    config = yaml.safe_load((ROOT / "configs/ours.yaml").read_text(encoding="utf-8"))
    start = time.perf_counter()
    metrics, predictions = evaluate_models(MODELS, config=config, protocol=protocol, split=args.split)
    elapsed = time.perf_counter() - start

    require(set(metrics["model"]) == set(MODELS) and len(metrics) == 2,
            "expected one metric row for each frozen candidate and comparator")
    target_dates = pd.to_datetime(predictions["target_date"])
    require(target_dates.min() >= expected_start and target_dates.max() < expected_end,
            "predictions fall outside the locked held-out period")
    require(not predictions.duplicated(["model", "target_date"]).any(), "duplicate model/date prediction rows")
    require(set(predictions["model"]) == set(MODELS), "prediction file is missing one of the two models")

    metrics.to_csv(out_dir / "heldout_fold_metrics.csv", index=False, float_format="%.10g")
    predictions.to_csv(out_dir / "heldout_predictions.csv", index=False, float_format="%.10g")
    summary = summarize_metrics(metrics)
    summary.to_csv(out_dir / "heldout_summary.csv", index=False, float_format="%.10g")
    run = {
        "status": "completed",
        "split": args.split,
        "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "frozen_commit": freeze["frozen_commit"],
        "config_sha256": freeze["file_sha256"]["configs/ours.yaml"],
        "models": MODELS,
        "period_start_inclusive": expected_start.strftime("%Y-%m-%d"),
        "period_end_exclusive": expected_end.strftime("%Y-%m-%d"),
        "fold_year": int(metrics["fold_year"].iloc[0]),
        "forecast_rows_per_model": int(predictions.groupby("model").size().iloc[0]),
        "primary_metric": "MCC at fixed threshold 0.5",
        "wall_seconds": elapsed,
        "prediction_file": str((out_dir / "heldout_predictions.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"Heldout_Evaluation=PASS split={args.split} rows_per_model={run['forecast_rows_per_model']} wall_seconds={elapsed:.3f}")
    for _, row in metrics.iterrows():
        print(f"{row['model']}_mcc={float(row['mcc']):.10f} auroc={float(row['auroc']):.10f}")


if __name__ == "__main__":
    main()
