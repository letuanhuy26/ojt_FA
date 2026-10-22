from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import FEATURE_COLUMNS, build_examples


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed SPY date-t features and next-session labels.")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/SPY_yfinance_1993-2025.csv")
    args = parser.parse_args()
    raw_path = args.raw if args.raw.is_absolute() else ROOT / args.raw
    if not raw_path.exists():
        raise SystemExit(f"Raw data not found: {raw_path}. Run scripts/download_data.py first.")
    raw = pd.read_csv(raw_path, parse_dates=["Date"])
    examples = build_examples(raw)
    out = ROOT / "data/processed/SPY_features_1993-2025.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(out, index=False, date_format="%Y-%m-%d", float_format="%.10g")
    manifest_path = ROOT / "data/manifests/SPY_download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_manifest = {
        "raw_file": str(raw_path.relative_to(ROOT)).replace("\\", "/") if raw_path.is_relative_to(ROOT) else raw_path.name,
        "raw_sha256": sha256(raw_path),
        "feature_file": str(out.relative_to(ROOT)).replace("\\", "/"),
        "feature_sha256": sha256(out),
        "rows": int(len(examples)),
        "first_feature_date": examples["date"].min().strftime("%Y-%m-%d"),
        "last_target_date": examples["target_date"].max().strftime("%Y-%m-%d"),
        "feature_columns": FEATURE_COLUMNS,
        "target": "next observed session adjusted close / adjusted open; zero is class 0",
        "raw_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }
    (ROOT / "data/manifests/SPY_features_manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    print(json.dumps(feature_manifest, indent=2))


if __name__ == "__main__":
    main()
