from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"GUARD_FAIL: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard data, chronology, and held-out periods.")
    parser.add_argument("--model-campaign", action="store_true")
    args = parser.parse_args()
    protocol_path = ROOT / "configs/protocol.yaml"
    lock_path = ROOT / "data/manifests/protocol.sha256"
    require(protocol_path.exists() and lock_path.exists(), "locked protocol or its SHA-256 file is missing")
    expected_protocol_hash = lock_path.read_text(encoding="ascii").strip()
    require(sha256(protocol_path) == expected_protocol_hash, "locked protocol.yaml changed during evaluation")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    evaluation = protocol["evaluation"]
    dev_years = evaluation["development_years"]
    require(dev_years == list(range(dev_years[0], dev_years[-1] + 1)), "development folds must be consecutive calendar years")
    dev_end = pd.Timestamp(f"{dev_years[-1] + 1}-01-01")
    confirm_start = pd.Timestamp(evaluation["confirmation_start"])
    final_start = pd.Timestamp(evaluation["final_start"])
    require(dev_end <= confirm_start < final_start, "development, confirmation, and final periods overlap or are out of order")
    require(evaluation["threshold"] == 0.5, "primary threshold changed without a new protocol")
    require(protocol["target"]["zero_return_class"] == "negative", "zero-return rule changed without a new protocol")

    raw_manifest_path = ROOT / "data/manifests/SPY_download_manifest.json"
    feature_manifest_path = ROOT / "data/manifests/SPY_features_manifest.json"
    require(raw_manifest_path.exists() and feature_manifest_path.exists(), "data audit manifests are missing")
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    raw_path = ROOT / raw_manifest["file"]
    feature_path = ROOT / feature_manifest["feature_file"]
    require(raw_path.exists() and feature_path.exists(), "local pilot data files are missing")
    require(sha256(raw_path) == raw_manifest["sha256"], "raw SPY file hash differs from the download manifest")
    require(sha256(raw_path) == feature_manifest["raw_sha256"], "feature table was not built from the current raw SPY file")
    require(sha256(feature_path) == feature_manifest["feature_sha256"], "feature table hash differs from its manifest")

    cache_manifest_path = ROOT / "data/manifests/regime_cache_manifest.json"
    require(cache_manifest_path.exists(), "development-fold causal-regime cache manifest is missing")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    require(set(cache_manifest.get("folds", {})) == {str(y) for y in dev_years}, "regime cache must contain exactly the predefined development folds")
    for year in dev_years:
        record = cache_manifest["folds"][str(year)]
        cache_path = ROOT / record["file"]
        require(cache_path.exists(), f"cached regime probabilities missing for fold {year}")
        require(sha256(cache_path) == record["sha256"], f"cached regime probabilities were modified for fold {year}")
        require(int(record["fold_year"]) == int(year), f"invalid fold marker for cached regime probabilities in {year}")
        require(record["filter_end_exclusive"] == f"{year + 1}-01-01", f"cached regime filter crosses the development fold boundary for {year}")
        require(int(record["last_filter_target_year"]) == int(year), f"cached regime filter reaches outside development year {year}")
        require(int(record["filter_rows"]) > int(record["training_rows"]), f"fold {year} cache has no validation observations")

    tracked_data = subprocess.run(
        ["git", "ls-files", "--", "data/raw", "data/processed"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(not tracked_data, "raw or derived price data must not be committed")
    split_bounds = {
        "development": (pd.Timestamp(evaluation["train_start"]), confirm_start),
        "confirmation": (confirm_start, pd.Timestamp(evaluation["confirmation_end_exclusive"])),
        "final": (final_start, pd.Timestamp(evaluation["final_end_exclusive"])),
    }
    prediction_files = []
    for split, (start, end) in split_bounds.items():
        split_dir = ROOT / "results" / split
        files = list(split_dir.glob("*predictions*.csv")) if split_dir.exists() else []
        run_path = split_dir / "run.json"
        if split != "development" and split_dir.exists():
            require(run_path.exists(), f"{split} results directory exists without a completed run manifest")
            run_record = json.loads(run_path.read_text(encoding="utf-8"))
            require(run_record.get("status") == "completed" and run_record.get("split") == split,
                    f"invalid or incomplete run manifest for {split}")
            if split in {"confirmation", "final"}:
                freeze_path = ROOT / "autoresearch/heldout-20261004-01/freeze.json"
                require(freeze_path.exists(), "held-out results exist without a candidate freeze manifest")
                freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
                require(run_record.get("frozen_commit") == freeze.get("frozen_commit"),
                        f"{split} result does not match the frozen candidate commit")
        if split == "final" and files:
            confirmation_run = ROOT / "results/confirmation/run.json"
            require(confirmation_run.exists(), "final results exist without a completed confirmation run")
        for prediction_file in files:
            predictions = pd.read_csv(prediction_file, parse_dates=["target_date"])
            require(not predictions.empty, f"empty prediction file: {prediction_file.name}")
            require(predictions["target_date"].min() >= start, f"pre-{split} date found in {prediction_file.name}")
            require(predictions["target_date"].max() < end, f"post-{split} date found in {prediction_file.name}")
            prediction_files.append(prediction_file)

    if args.model_campaign:
        scope_path = ROOT / "autoresearch/model-20261004-01/scope-start.json"
        require(scope_path.exists(), "model campaign scope-start checkpoint is missing")
        base = json.loads(scope_path.read_text(encoding="utf-8"))["baseline_commit"]
        changed = subprocess.run(
            ["git", "diff", "--name-only", f"{base}..HEAD"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        allowed = {"src/models/ours.py", "configs/ours.yaml"}
        allowed_log_prefix = "autoresearch/model-20261004-01/"
        disallowed = [path for path in changed if path not in allowed and not path.startswith(allowed_log_prefix)]
        require(not disallowed, f"model campaign changed files outside scope: {disallowed}")

    print("Protocol_Guard=PASS")
    print(f"Data rows: raw={raw_manifest['rows']}; features={feature_manifest['rows']}")
    print(f"Development folds: {dev_years}; HMM caches verified={len(dev_years)}; prediction files checked={len(prediction_files)}")


if __name__ == "__main__":
    main()
