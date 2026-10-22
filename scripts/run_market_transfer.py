from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.evaluation.walkforward as walkforward
from src.data.features import build_examples
from src.regimes.hmm import DiagonalGaussianHMM


CONFIG_PATH = ROOT / "configs/market_transfer.yaml"
PILOT_PROTOCOL_PATH = ROOT / "configs/protocol.yaml"
MODEL_CONFIG_PATH = ROOT / "configs/ours.yaml"
START = "1993-01-01"
END_EXCLUSIVE = "2026-01-01"
MODELS = [
    "regime_dynamic_gate_lstm",
    "dynamic_gate_lstm",
    "regime_as_feature_lstm",
    "random_forest",
]
SPLITS = ["development", "confirmation", "final"]
BLOCK_LENGTH = 20
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 1729


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_or_load(symbol: str) -> tuple[pd.DataFrame, dict]:
    raw_path = ROOT / f"data/raw/{symbol}_yfinance_1993-2025.csv"
    manifest_path = ROOT / f"data/manifests/{symbol}_transfer_download_manifest.json"
    if raw_path.exists():
        raw = pd.read_csv(raw_path, parse_dates=["Date"])
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        elif symbol == "SPY":
            original_path = ROOT / "data/manifests/SPY_download_manifest.json"
            manifest = json.loads(original_path.read_text(encoding="utf-8"))
        else:
            manifest = {}
    else:
        bars = yf.Ticker(symbol).history(
            start=START,
            end=END_EXCLUSIVE,
            interval="1d",
            auto_adjust=False,
            actions=True,
            repair=False,
            keepna=True,
        )
        if bars is None or bars.empty:
            raise RuntimeError(f"Yahoo Finance returned no rows for {symbol}.")
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)
        bars.index = pd.to_datetime(bars.index)
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        bars.index.name = "Date"
        raw = bars.reset_index()
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.normalize()
        raw = raw.sort_values("Date").reset_index(drop=True)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_path, index=False, date_format="%Y-%m-%d", float_format="%.10g")
        manifest = {
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "provider": "Yahoo Finance historical prices via yfinance",
            "yfinance_version": yf.__version__,
            "python_version": platform.python_version(),
            "request": {"start_inclusive": START, "end_exclusive": END_EXCLUSIVE,
                        "interval": "1d", "auto_adjust": False, "actions": True},
            "file": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
            "rights_status": "local development only; publication and redistribution rights unresolved",
        }

    required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    if not required.issubset(raw.columns):
        raise RuntimeError(f"{symbol} data missing columns: {sorted(required - set(raw.columns))}")
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.normalize()
    raw = raw.sort_values("Date").reset_index(drop=True)
    if raw.empty or raw["Date"].duplicated().any():
        raise RuntimeError(f"Empty or duplicate dates in {symbol} data.")
    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    if raw[price_cols].isna().any().any() or raw["Volume"].isna().any():
        raise RuntimeError(f"Missing price or volume values in {symbol} response; do not silently fill.")
    manifest.update({
        "symbol": symbol,
        "file": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(raw_path),
        "rows": int(len(raw)),
        "first_session": raw["Date"].iloc[0].strftime("%Y-%m-%d"),
        "last_session": raw["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "columns": list(raw.columns),
        "missing_by_column": {str(k): int(v) for k, v in raw.isna().sum().items()},
        "rights_status": "local development only; publication and redistribution rights unresolved",
    })
    if symbol != "SPY" or not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return raw, manifest


def uncached_fold_regime_posteriors(year: int, start: pd.Timestamp, end: pd.Timestamp,
                                    z_raw: np.ndarray, train_mask: np.ndarray,
                                    protocol: dict) -> tuple[np.ndarray, dict]:
    """Fit/filter regimes for another asset without touching the SPY cache."""
    regime = protocol["regime"]
    seed = int(protocol["evaluation"]["random_seed"])
    scaler = StandardScaler().fit(z_raw[train_mask])
    z_scaled = scaler.transform(z_raw)
    model = DiagonalGaussianHMM(
        n_states=int(regime["states"]),
        max_iter=int(regime["em_iterations"]),
        variance_floor=float(regime["covariance_floor"]),
        random_state=seed,
    ).fit(z_scaled[train_mask])
    posterior = model.filter(z_scaled)
    details = {
        "log_likelihood_per_obs": float(model.log_likelihood_ / int(train_mask.sum())),
        "iterations": int(model.n_iter_),
        "training_rows": int(train_mask.sum()),
        "filter_rows": int(len(z_raw)),
    }
    return posterior, details


def paired_circular_block_bootstrap(candidate: pd.DataFrame,
                                    baseline: pd.DataFrame) -> dict:
    candidate = candidate.sort_values("target_date").reset_index(drop=True)
    baseline = baseline.sort_values("target_date").reset_index(drop=True)
    if candidate["target_date"].tolist() != baseline["target_date"].tolist():
        raise ValueError("Candidate and baseline do not share target dates.")
    y = candidate["y_true"].to_numpy(dtype=np.int8)
    if not np.array_equal(y, baseline["y_true"].to_numpy(dtype=np.int8)):
        raise ValueError("Candidate and baseline labels do not match.")
    candidate_prediction = (candidate["probability"].to_numpy(dtype=float) >= 0.5).astype(np.int8)
    baseline_prediction = (baseline["probability"].to_numpy(dtype=float) >= 0.5).astype(np.int8)
    point = matthews_corrcoef(y, candidate_prediction) - matthews_corrcoef(y, baseline_prediction)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    offsets = np.arange(BLOCK_LENGTH)
    n = len(y)
    n_blocks = int(np.ceil(n / BLOCK_LENGTH))
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        starts = rng.integers(0, n, size=n_blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        samples[replicate] = (
            matthews_corrcoef(y[indices], candidate_prediction[indices])
            - matthews_corrcoef(y[indices], baseline_prediction[indices])
        )
    return {
        "n": int(n),
        "point_delta_mcc": float(point),
        "bootstrap_block_sessions": BLOCK_LENGTH,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci95_percentile_low": float(np.quantile(samples, 0.025)),
        "ci95_percentile_high": float(np.quantile(samples, 0.975)),
        "p_value": None,
    }


def main() -> None:
    print("[autoresearch] mode: classic")
    print("Goal: evaluate frozen SPY h=1 hyperparameter transfer to UK and Japan ETF proxies.")
    print("Metric: mean development MCC; later-block paired MCC differences with circular-block intervals.")
    print("Verify: input bars, target alignment, same forecast origins, within-asset temporal fitting.")
    print("Guard: no hyperparameter tuning; no original SPY HMM cache writes; raw data local-only.")
    print("Iterations: fixed-config evaluation across 2 external markets × 3 splits × 4 models.")

    config = walkforward.load_yaml(CONFIG_PATH)
    pilot_protocol = walkforward.load_yaml(PILOT_PROTOCOL_PATH)
    model_config = walkforward.load_yaml(MODEL_CONFIG_PATH)
    if [market["symbol"] for market in config["markets"]] != ["EWU", "EWJ"]:
        raise SystemExit("Unexpected market proxy list in config.")
    if config["models"]["compared"] != MODELS:
        raise SystemExit("Model list differs from the written market-transfer configuration.")
    if config["evaluation"]["development_years"] != pilot_protocol["evaluation"]["development_years"]:
        raise SystemExit("Market-transfer development folds differ from the source protocol.")
    for key in ["confirmation_start", "confirmation_end_exclusive", "final_start", "final_end_exclusive"]:
        if config["evaluation"][key] != pilot_protocol["evaluation"][key]:
            raise SystemExit(f"Market-transfer split mismatch for {key}.")

    out_dir = ROOT / config["outputs"]["root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    protected = [out_dir / "fold_metrics.csv", out_dir / "predictions.csv",
                 out_dir / "heldout_mcc_bootstrap.csv", out_dir / "run.json"]
    if any(path.exists() for path in protected):
        raise SystemExit("Refusing to overwrite existing market-transfer outputs.")
    processed_dir = ROOT / "data/processed/market_transfer_20261004"
    processed_dir.mkdir(parents=True, exist_ok=True)
    protocol = copy.deepcopy(pilot_protocol)
    protocol["status"] = "exploratory_cross_market_configuration_transfer"
    walkforward.fold_regime_posteriors = uncached_fold_regime_posteriors

    market_metadata = []
    metrics_all, predictions_all = [], []
    for market in config["markets"]:
        symbol = market["symbol"]
        raw, manifest = download_or_load(symbol)
        examples = build_examples(raw)
        if len(examples) < 3000:
            raise SystemExit(f"Insufficient history for {symbol}: {len(examples)} examples.")
        data_path = processed_dir / f"{symbol}_features.csv"
        examples.to_csv(data_path, index=False, float_format="%.12g")
        protocol["market"] = {"symbol": symbol, "description": market["instrument"]}
        market_metadata.append({
            "region": market["region"],
            "symbol": symbol,
            "instrument": market["instrument"],
            "role": market["role"],
            "raw_sha256": manifest["sha256"],
            "rows": manifest["rows"],
            "first_session": manifest["first_session"],
            "last_session": manifest["last_session"],
            "feature_rows": int(len(examples)),
            "feature_file": str(data_path.relative_to(ROOT)).replace("\\", "/"),
        })
        for split in SPLITS:
            metrics, predictions = walkforward.evaluate_models(
                MODELS,
                config=model_config,
                protocol=protocol,
                data_path=data_path,
                split=split,
            )
            metrics = metrics.assign(region=market["region"], symbol=symbol, split=split)
            predictions = predictions.assign(region=market["region"], symbol=symbol, split=split)
            years = [2015, 2016, 2017, 2018, 2019] if split == "development" else ([2020] if split == "confirmation" else [2022])
            expected_rows = [year for year in years for _ in MODELS]
            if metrics["fold_year"].astype(int).tolist() != expected_rows:
                raise SystemExit(f"Unexpected {symbol}/{split} folds: {metrics['fold_year'].tolist()}")
            if set(metrics["model"]) != set(MODELS) or not np.isfinite(metrics["mcc"].to_numpy(dtype=float)).all():
                raise SystemExit(f"Invalid metrics in {symbol}/{split}.")
            for year in years:
                year_predictions = predictions[predictions["fold_year"] == year]
                date_sets = [
                    tuple(year_predictions.loc[year_predictions["model"] == model, "target_date"].astype(str))
                    for model in MODELS
                ]
                if any(dates != date_sets[0] for dates in date_sets[1:]):
                    raise SystemExit(f"Forecast origins differ across models for {symbol}/{split}/{year}.")
            metrics_all.append(metrics)
            predictions_all.append(predictions)
            print(f"{symbol} {split}: {len(metrics)} model-fold rows; n={int(metrics['n'].sum())}", flush=True)

    metrics_frame = pd.concat(metrics_all, ignore_index=True)
    predictions_frame = pd.concat(predictions_all, ignore_index=True)
    metrics_path = out_dir / "fold_metrics.csv"
    predictions_path = out_dir / "predictions.csv"
    bootstrap_path = out_dir / "heldout_mcc_bootstrap.csv"
    metrics_frame.to_csv(metrics_path, index=False, float_format="%.10g")
    predictions_frame.to_csv(predictions_path, index=False, float_format="%.10g")

    bootstrap_rows = []
    for market in config["markets"]:
        symbol = market["symbol"]
        for split in ["confirmation", "final"]:
            selected = predictions_frame[
                (predictions_frame["symbol"] == symbol) & (predictions_frame["split"] == split)
            ]
            candidate = selected[selected["model"] == "regime_dynamic_gate_lstm"]
            baseline = selected[selected["model"] == "random_forest"]
            bootstrap_rows.append({
                "region": market["region"], "symbol": symbol, "split": split,
                **paired_circular_block_bootstrap(candidate, baseline),
            })
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_frame.to_csv(bootstrap_path, index=False, float_format="%.10g")

    metadata = {
        "run_id": config["outputs"]["autoresearch_log"].split("/")[-1],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256(CONFIG_PATH),
        "source_protocol": str(PILOT_PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source_protocol_sha256": sha256(PILOT_PROTOCOL_PATH),
        "frozen_model_config": str(MODEL_CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        "frozen_model_config_sha256": sha256(MODEL_CONFIG_PATH),
        "provider": "Yahoo Finance historical prices via yfinance",
        "yfinance_version": yf.__version__,
        "markets": market_metadata,
        "models": MODELS,
        "splits": SPLITS,
        "tuning": "none; h=1 SPY configuration transferred unchanged; model weights refit within each market",
        "heldout_accessed": True,
        "interpretation": "posthoc exploratory cross-market configuration transfer; same source calendar blocks as SPY pilot",
        "rights_status": "unresolved; raw and derived data are local-only and excluded from Git",
        "economics": "not evaluated",
        "outputs": {
            "fold_metrics": str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
            "predictions_local": str(predictions_path.relative_to(ROOT)).replace("\\", "/"),
            "heldout_mcc_bootstrap": str(bootstrap_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved market-transfer outputs in {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
