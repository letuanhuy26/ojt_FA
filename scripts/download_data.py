from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_DIR = ROOT / "data" / "manifests"
SYMBOL = "SPY"
START = "1993-01-01"
END_EXCLUSIVE = "2026-01-01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local SPY daily bars for development.")
    parser.add_argument("--refresh", action="store_true", help="save a new timestamped copy if the default file exists")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RAW_DIR / f"{SYMBOL}_yfinance_1993-2025.csv"
    if path.exists():
        if not args.refresh:
            raise SystemExit(f"Refusing to overwrite immutable raw file: {path}. Use --refresh for a timestamped copy.")
        path = RAW_DIR / f"{SYMBOL}_yfinance_1993-2025_{stamp}.csv"

    bars = yf.Ticker(SYMBOL).history(
        start=START,
        end=END_EXCLUSIVE,
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=False,
        keepna=True,
    )
    if bars is None or bars.empty:
        raise RuntimeError("Yahoo Finance returned no rows.")
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)
    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "Date"
    bars = bars.reset_index()
    bars["Date"] = pd.to_datetime(bars["Date"]).dt.normalize()
    bars = bars.sort_values("Date").reset_index(drop=True)

    required = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    absent = sorted(set(required) - set(bars.columns))
    if absent:
        raise RuntimeError(f"Yahoo response missing required columns: {absent}; returned={list(bars.columns)}")
    if bars["Date"].duplicated().any():
        raise RuntimeError("Duplicate session dates in source response.")
    if not bars[required[1:6]].notna().all().all():
        raise RuntimeError("Missing OHLC/adjusted-close observations; do not silently impute raw bars.")

    bars.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.10g")
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "provider": "Yahoo Finance historical prices via yfinance",
        "yfinance_version": yf.__version__,
        "python_version": platform.python_version(),
        "request": {"start_inclusive": START, "end_exclusive": END_EXCLUSIVE, "interval": "1d", "auto_adjust": False, "actions": True},
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(path),
        "rows": int(len(bars)),
        "first_session": bars["Date"].iloc[0].strftime("%Y-%m-%d"),
        "last_session": bars["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "columns": list(bars.columns),
        "missing_by_column": {str(k): int(v) for k, v in bars.isna().sum().items()},
        "rights_status": "local development only; publication and redistribution rights unresolved",
    }
    manifest_path = MANIFEST_DIR / f"{SYMBOL}_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
