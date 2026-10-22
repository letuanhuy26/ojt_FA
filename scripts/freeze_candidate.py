from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "autoresearch/heldout-20261004-01/freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FREEZE_FAIL: {message}")


def main() -> None:
    require(not FREEZE_PATH.exists(), "candidate freeze already exists")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(not status, "working tree must be clean before candidate freeze")

    config_path = ROOT / "configs/ours.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        "window": 60,
        "hidden_size": 32,
        "gate_hidden_size": 32,
        "dropout": 0.0,
        "epochs": 20,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "lambda_sparsity": 0.005,
        "lambda_stability": 0.01,
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"selected config field {key} differs from the retained development candidate")

    ledger_path = ROOT / "autoresearch/model-20261004-01/results.tsv"
    with ledger_path.open("r", encoding="utf-8", newline="") as stream:
        ledger = list(csv.DictReader(stream, delimiter="\t"))
    iterations = {int(row["iteration"]) for row in ledger if row["iteration"].isdigit() and int(row["iteration"]) > 0}
    require(iterations == set(range(1, 11)), "the ten-iteration development ledger is incomplete")
    keep_scores = [
        float(row["mean_development_mcc"])
        for row in ledger
        if row["decision"] in {"INCUMBENT", "KEEP"} and row["mean_development_mcc"]
    ]
    require(keep_scores and abs(max(keep_scores) - 0.0331907033) < 1e-9,
            "best retained development result differs from the campaign record")

    raw_manifest = json.loads((ROOT / "data/manifests/SPY_download_manifest.json").read_text(encoding="utf-8"))
    feature_manifest = json.loads((ROOT / "data/manifests/SPY_features_manifest.json").read_text(encoding="utf-8"))
    files = [
        "configs/ours.yaml",
        "configs/protocol.yaml",
        "configs/heldout_analysis.yaml",
        "requirements.txt",
        "data/manifests/protocol.sha256",
        "src/models/ours.py",
        "src/regimes/hmm.py",
        "src/data/features.py",
        "src/evaluation/walkforward.py",
        "src/evaluation/regime_cache.py",
        "scripts/freeze_candidate.py",
        "scripts/run_heldout.py",
        "scripts/analyze_heldout.py",
        "scripts/check_protocol.py",
        raw_manifest["file"],
        feature_manifest["feature_file"],
    ]
    hashes = {name: sha256(ROOT / name) for name in files}
    locked_protocol_hash = (ROOT / "data/manifests/protocol.sha256").read_text(encoding="ascii").strip()
    require(hashes["configs/protocol.yaml"] == locked_protocol_hash,
            "protocol hash lock does not match protocol.yaml")

    for split in ("confirmation", "final"):
        result_dir = ROOT / "results" / split
        require(not result_dir.exists(), f"{split} output directory already exists")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    payload = {
        "status": "candidate_frozen_before_confirmation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_commit": head,
        "market": "SPY (provisional single-ETF pilot)",
        "selected_model": config["model"],
        "selected_config": config,
        "selected_development_mean_mcc": max(keep_scores),
        "development_campaign": "autoresearch/model-20261004-01/results.tsv",
        "confirmation_period": ["2020-01-01", "2021-12-31"],
        "final_period": ["2022-01-01", "2025-12-31"],
        "run_policy": "one confirmation evaluation, then one final evaluation; no post-confirmation tuning",
        "file_sha256": hashes,
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Candidate_Freeze=PASS commit={head}")
    print(f"Frozen_config_sha256={hashes['configs/ours.yaml']}")


if __name__ == "__main__":
    main()
