from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.evaluation.walkforward as walkforward
from src.data.features import build_examples
from src.regimes.hmm import DiagonalGaussianHMM


CONFIG_PATH = ROOT / "configs/horizon_transfer.yaml"
RAW_PATH = ROOT / "data/raw/SPY_yfinance_1993-2025.csv"
PILOT_PROTOCOL_PATH = ROOT / "configs/protocol.yaml"
MODEL_CONFIG_PATH = ROOT / "configs/ours.yaml"
HORIZONS = [5, 20]
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


def target_frame(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Use the pilot's causal features and build a target ending h sessions ahead."""
    features = build_examples(raw)
    dates = pd.to_datetime(raw["Date"], utc=True).dt.tz_convert(None).dt.normalize()
    close = pd.to_numeric(raw["Close"], errors="coerce").to_numpy(dtype=float)
    adj_close = pd.to_numeric(raw["Adj Close"], errors="coerce").to_numpy(dtype=float)
    adj_open = pd.to_numeric(raw["Open"], errors="coerce").to_numpy(dtype=float) * (adj_close / close)
    date_to_position = {pd.Timestamp(date): pos for pos, date in enumerate(dates)}
    origin_positions = features["date"].map(date_to_position)
    if origin_positions.isna().any():
        raise ValueError("Feature dates failed to align with raw market dates.")
    origin_positions = origin_positions.to_numpy(dtype=int)
    end_positions = origin_positions + int(horizon)
    valid = end_positions < len(raw)
    frame = features.loc[valid].copy().reset_index(drop=True)
    origin_positions = origin_positions[valid]
    end_positions = end_positions[valid]
    entry_positions = origin_positions + 1
    returns = np.log(adj_close[end_positions] / adj_open[entry_positions])
    frame["target_date"] = dates.iloc[end_positions].to_numpy()
    frame["target_return"] = returns
    frame["target"] = (returns > 0).astype(np.int8)
    if not np.isfinite(returns).all():
        raise ValueError(f"Non-finite target returns for h={horizon}.")
    return frame


def uncached_fold_regime_posteriors(year: int, start: pd.Timestamp, end: pd.Timestamp,
                                    z_raw: np.ndarray, train_mask: np.ndarray,
                                    protocol: dict) -> tuple[np.ndarray, dict]:
    """Fit/filter regimes for the extension without reusing the h=1 cache."""
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
    p_candidate = candidate["probability"].to_numpy(dtype=float)
    p_baseline = baseline["probability"].to_numpy(dtype=float)
    yhat_candidate = (p_candidate >= 0.5).astype(np.int8)
    yhat_baseline = (p_baseline >= 0.5).astype(np.int8)
    point_delta = matthews_corrcoef(y, yhat_candidate) - matthews_corrcoef(y, yhat_baseline)
    n = len(y)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    offsets = np.arange(BLOCK_LENGTH)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    n_blocks = int(np.ceil(n / BLOCK_LENGTH))
    for replicate in range(BOOTSTRAP_REPLICATES):
        starts = rng.integers(0, n, size=n_blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        deltas[replicate] = (
            matthews_corrcoef(y[indices], yhat_candidate[indices])
            - matthews_corrcoef(y[indices], yhat_baseline[indices])
        )
    return {
        "n": int(n),
        "metric": "MCC(candidate)-MCC(random_forest)",
        "point_delta_mcc": float(point_delta),
        "bootstrap_block_sessions": BLOCK_LENGTH,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci95_percentile_low": float(np.quantile(deltas, 0.025)),
        "ci95_percentile_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_unit": "paired circular blocks over ordered forecast origins",
        "p_value": None,
    }


def main() -> None:
    print("[autoresearch] mode: classic")
    print("Goal: test transfer of the frozen h=1 candidate configuration to h=5 and h=20 targets.")
    print("Metric: mean fold MCC; held-out paired MCC differences are descriptive moving-block intervals.")
    print("Verify: target alignment, temporal split, shared forecast dates, finite metrics.")
    print("Guard: separate extension config; no horizon-specific tuning; no h=1 cache writes.")
    print("Iterations: fixed-config evaluation across 2 horizons × 3 splits × 4 models.")

    config = walkforward.load_yaml(CONFIG_PATH)
    pilot_protocol = walkforward.load_yaml(PILOT_PROTOCOL_PATH)
    if config["target"]["horizons_sessions"] != HORIZONS or config["models"]["compared"] != MODELS:
        raise SystemExit("Script constants do not match the written horizon-transfer configuration.")
    expected_ranges = {
        "development_years": [2015, 2016, 2017, 2018, 2019],
        "confirmation_start": "2020-01-01",
        "confirmation_end_exclusive": "2022-01-01",
        "final_start": "2022-01-01",
        "final_end_exclusive": "2026-01-01",
    }
    for key, expected in expected_ranges.items():
        if pilot_protocol["evaluation"][key] != expected:
            raise SystemExit(f"Pilot split mismatch for {key}: {pilot_protocol['evaluation'][key]!r}")
    protocol = copy.deepcopy(pilot_protocol)
    protocol["status"] = "exploratory_cross_horizon_transfer"
    protocol["target"] = {
        "name": "open_to_close_h_session_return_direction",
        "definition": "1[log(Adj_Close_(t+h) / Adj_Open_(t+1)) > 0]",
        "signal_time": "after_close_t",
        "execution_reference": "open_t_plus_1",
        "target_end": "close_t_plus_h",
        "zero_return_class": "negative",
        "missing_target": "drop",
    }
    protocol["evaluation"]["economics"] = "not_evaluated_due_to_overlapping_horizon_positions"
    model_config = walkforward.load_yaml(MODEL_CONFIG_PATH)
    raw = pd.read_csv(RAW_PATH, parse_dates=["Date"])
    if len(raw) != 8288:
        raise SystemExit(f"Unexpected raw SPY row count: {len(raw)}")
    pilot_examples = build_examples(raw)
    one_session_examples = target_frame(raw, 1)
    if pilot_examples["date"].tolist() != one_session_examples["date"].tolist():
        raise SystemExit("h=1 alignment check failed: origin dates differ from the frozen feature table.")
    if pilot_examples["target_date"].tolist() != one_session_examples["target_date"].tolist():
        raise SystemExit("h=1 alignment check failed: target dates differ from the frozen feature table.")
    if not np.allclose(
        pilot_examples["target_return"].to_numpy(dtype=float),
        one_session_examples["target_return"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    ):
        raise SystemExit("h=1 alignment check failed: target returns differ from the frozen feature table.")

    out_dir = ROOT / config["outputs"]["root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    protected_outputs = [
        out_dir / "fold_metrics.csv",
        out_dir / "predictions.csv",
        out_dir / "heldout_mcc_bootstrap.csv",
        out_dir / "run.json",
    ]
    if any(path.exists() for path in protected_outputs):
        raise SystemExit("Refusing to overwrite an existing completed horizon-transfer result set.")
    target_cache_dir = ROOT / "data/processed/horizon_transfer_20261004"
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    all_metrics, all_predictions = [], []

    # Patch the module-level function used by prepare_fold so this extension
    # never overwrites the original one-session HMM cache or its manifest.
    walkforward.fold_regime_posteriors = uncached_fold_regime_posteriors

    for horizon in HORIZONS:
        frame = target_frame(raw, horizon)
        target_path = target_cache_dir / f"SPY_features_h{horizon}.csv"
        frame.to_csv(target_path, index=False, float_format="%.12g")
        for split in SPLITS:
            metrics, predictions = walkforward.evaluate_models(
                MODELS,
                config=model_config,
                protocol=protocol,
                data_path=target_path,
                split=split,
            )
            metrics = metrics.assign(horizon_sessions=horizon, split=split)
            predictions = predictions.assign(horizon_sessions=horizon, split=split)
            expected_years = [2015, 2016, 2017, 2018, 2019] if split == "development" else ([2020] if split == "confirmation" else [2022])
            expected_rows = [year for year in expected_years for _ in MODELS]
            if metrics["fold_year"].astype(int).tolist() != expected_rows:
                raise SystemExit(f"Unexpected {split}/h={horizon} folds: {metrics['fold_year'].tolist()}")
            if not np.isfinite(metrics["mcc"].to_numpy(dtype=float)).all():
                raise SystemExit(f"Non-finite MCC in {split}/h={horizon}")
            expected_models = set(MODELS)
            if set(metrics["model"]) != expected_models or len(metrics) != len(expected_years) * len(MODELS):
                raise SystemExit(f"Model/fold count mismatch in {split}/h={horizon}")
            # Verify the label-purge rule by construction: all fitting labels
            # are selected by target_date < validation start in prepare_fold.
            all_metrics.append(metrics)
            all_predictions.append(predictions)
            print(f"h={horizon} split={split}: {len(metrics)} model-fold rows, n={int(metrics['n'].sum())}", flush=True)

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    metrics_path = out_dir / "fold_metrics.csv"
    predictions_path = out_dir / "predictions.csv"
    metrics_frame.to_csv(metrics_path, index=False, float_format="%.10g")
    predictions_frame.to_csv(predictions_path, index=False, float_format="%.10g")

    bootstrap_rows = []
    heldout_summaries = []
    for horizon in HORIZONS:
        for split in ["confirmation", "final"]:
            current = predictions_frame[
                (predictions_frame["horizon_sessions"] == horizon)
                & (predictions_frame["split"] == split)
            ]
            candidate = current[current["model"] == "regime_dynamic_gate_lstm"]
            baseline = current[current["model"] == "random_forest"]
            result = paired_circular_block_bootstrap(candidate, baseline)
            bootstrap_rows.append({"horizon_sessions": horizon, "split": split, **result})
            heldout_summaries.append({
                "horizon_sessions": horizon,
                "split": split,
                "n": result["n"],
                "candidate_mcc": float(metrics_frame.loc[
                    (metrics_frame["horizon_sessions"] == horizon)
                    & (metrics_frame["split"] == split)
                    & (metrics_frame["model"] == "regime_dynamic_gate_lstm"), "mcc"
                ].iloc[0]),
                "random_forest_mcc": float(metrics_frame.loc[
                    (metrics_frame["horizon_sessions"] == horizon)
                    & (metrics_frame["split"] == split)
                    & (metrics_frame["model"] == "random_forest"), "mcc"
                ].iloc[0]),
                "delta_mcc": result["point_delta_mcc"],
                "ci95_low": result["ci95_percentile_low"],
                "ci95_high": result["ci95_percentile_high"],
            })
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_path = out_dir / "heldout_mcc_bootstrap.csv"
    bootstrap_frame.to_csv(bootstrap_path, index=False, float_format="%.10g")

    metadata = {
        "run_id": config["outputs"]["autoresearch_log"].split("/")[-1],
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG_PATH),
        "source_protocol": str(PILOT_PROTOCOL_PATH.relative_to(ROOT)),
        "source_protocol_sha256": sha256(PILOT_PROTOCOL_PATH),
        "raw_data": str(RAW_PATH.relative_to(ROOT)),
        "raw_data_sha256": sha256(RAW_PATH),
        "horizons_sessions": HORIZONS,
        "models": MODELS,
        "splits": SPLITS,
        "development_only_tuning": "none; original frozen h=1 configuration transferred unchanged",
        "cross_horizon_status": "posthoc exploratory; does not modify the h=1 freeze or earlier h=1 held-out evaluation",
        "heldout_accessed": True,
        "economics": "not computed; overlapping h-session forecast targets are not standalone trades",
        "outputs": {
            "fold_metrics": str(metrics_path.relative_to(ROOT)),
            "predictions_local": str(predictions_path.relative_to(ROOT)),
            "heldout_mcc_bootstrap": str(bootstrap_path.relative_to(ROOT)),
        },
        "heldout_summary": heldout_summaries,
    }
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nHeld-out cross-horizon transfer summaries (exploratory):")
    print(pd.DataFrame(heldout_summaries).to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Saved outputs in {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
