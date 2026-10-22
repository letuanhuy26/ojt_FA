from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_strategy(frame: pd.DataFrame, position: np.ndarray, cost_bps: float, annual_sessions: int) -> dict:
    gross_simple = np.expm1(frame["target_return"].to_numpy(dtype=np.float64))
    rate = cost_bps / 10000.0
    # Entry costs are charged on opening notional; exit costs are charged on the
    # notional after the intraday gross return. The position is one or zero.
    factor = np.ones(len(frame), dtype=np.float64)
    active = position.astype(bool)
    factor[active] = (1.0 + gross_simple[active]) * (1.0 - rate) - rate
    if np.any(factor <= 0):
        raise SystemExit("Cost scenario produced a non-positive wealth multiplier.")
    daily_return = factor - 1.0
    equity = np.cumprod(factor)
    equity_with_start = np.r_[1.0, equity]
    drawdown = equity_with_start / np.maximum.accumulate(equity_with_start) - 1.0
    sd = float(np.std(daily_return, ddof=1)) if len(daily_return) > 1 else float("nan")
    n = len(frame)
    return {
        "n_sessions": n,
        "active_trades": int(active.sum()),
        "active_fraction": float(active.mean()),
        "gross_total_return": float(np.prod(1.0 + position * gross_simple) - 1.0),
        "net_total_return": float(equity[-1] - 1.0),
        "annualized_net_return": float(equity[-1] ** (annual_sessions / n) - 1.0),
        "annualized_sharpe_rf0": float(np.mean(daily_return) / sd * np.sqrt(annual_sessions)) if sd > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "annualized_two_way_turnover_multiple": float(2.0 * active.mean() * annual_sessions),
        "net_win_fraction_active": float((daily_return[active] > 0).mean()) if active.any() else float("nan"),
    }


def final_wealth_at_cost(frame: pd.DataFrame, position: np.ndarray, cost_bps: float) -> float:
    gross_simple = np.expm1(frame["target_return"].to_numpy(dtype=np.float64))
    rate = cost_bps / 10000.0
    factor = np.ones(len(frame), dtype=np.float64)
    active = position.astype(bool)
    factor[active] = (1.0 + gross_simple[active]) * (1.0 - rate) - rate
    return float(np.prod(factor))


def breakeven_side_cost_bps(frame: pd.DataFrame, position: np.ndarray) -> float:
    if final_wealth_at_cost(frame, position, 0.0) <= 1.0:
        return 0.0
    lo, hi = 0.0, 100.0
    if final_wealth_at_cost(frame, position, hi) > 1.0:
        return float("nan")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if final_wealth_at_cost(frame, position, mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def main() -> None:
    analysis_path = ROOT / "configs/economic_sensitivity.yaml"
    config = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    if config["split"] != "development":
        raise SystemExit("This exploratory economic analysis is restricted to development.")
    threshold = float(config["decision_threshold"])
    protocol_path = ROOT / "configs/protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if threshold != float(protocol["evaluation"]["threshold"]):
        raise SystemExit("Economic sensitivity threshold differs from the locked protocol.")
    freeze_path = ROOT / "autoresearch/heldout-20261004-01/freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "candidate_frozen_before_confirmation":
        raise SystemExit("Candidate freeze manifest is not in the expected state.")
    if freeze["selected_model"] != "regime_dynamic_gate_lstm":
        raise SystemExit("Unexpected frozen candidate model.")
    base_predictions = pd.read_csv(
        ROOT / "results/development/baseline_predictions.csv", parse_dates=["target_date"]
    )
    selected = pd.read_csv(
        ROOT / "results/development/gate_diagnostic_signal_series.csv",
        parse_dates=["signal_date", "target_date"],
    )
    selected = selected.loc[selected["model_seed_base"] == 1729,
                            ["fold_year", "signal_date", "target_date", "y_true", "probability"]]
    rf = base_predictions.loc[base_predictions["model"] == "random_forest",
                             ["fold_year", "target_date", "y_true", "probability"]]
    development_target_dates = set(selected["target_date"]) | set(rf["target_date"])
    feature_path = ROOT / "data/processed/SPY_features_1993-2025.csv"
    features = pd.read_csv(
        feature_path,
        parse_dates=["date", "target_date"],
        usecols=["date", "target_date", "target", "target_return"],
    )
    features = features.loc[features["target_date"].isin(development_target_dates)].copy()
    forecasts = {
        "selected_rc_sfg_seed_1729": selected,
        "random_forest": rf,
    }
    date_sets = []
    for name, predictions in forecasts.items():
        if predictions.duplicated(["fold_year", "target_date"]).any():
            raise SystemExit(f"Duplicate forecast origin in {name}.")
        aligned = predictions.merge(features, on="target_date", how="left", validate="one_to_one")
        if aligned["target_return"].isna().any() or not np.array_equal(
            aligned["y_true"].to_numpy(dtype=int), aligned["target"].to_numpy(dtype=int)
        ):
            raise SystemExit(f"Target/return alignment failed for {name}.")
        if "signal_date" in aligned and not (aligned["signal_date"].dt.normalize() == aligned["date"].dt.normalize()).all():
            raise SystemExit("Selected-candidate signal dates do not align to feature dates.")
        date_sets.append(set(aligned["target_date"]))
        forecasts[name] = aligned.sort_values("target_date").reset_index(drop=True)
    if date_sets[0] != date_sets[1]:
        raise SystemExit("Candidate and Random Forest do not share forecast origins.")
    fold_dates = {
        name: set(zip(frame["fold_year"].astype(int), frame["target_date"].dt.strftime("%Y-%m-%d")))
        for name, frame in forecasts.items()
    }
    if fold_dates["selected_rc_sfg_seed_1729"] != fold_dates["random_forest"]:
        raise SystemExit("Candidate and Random Forest do not share fold-year/target-date pairs.")
    if len(forecasts["selected_rc_sfg_seed_1729"]) != 1258:
        raise SystemExit("Unexpected development forecast count; expected five locked annual folds.")
    if sorted(forecasts["selected_rc_sfg_seed_1729"]["fold_year"].astype(int).unique()) != [2015, 2016, 2017, 2018, 2019]:
        raise SystemExit("Economic sensitivity must use only the five locked development folds.")

    all_rows: list[dict] = []
    fold_rows: list[dict] = []
    pooled: dict[str, pd.DataFrame] = {}
    for model_name, forecast in forecasts.items():
        forecast = forecast.sort_values("target_date").reset_index(drop=True)
        position = (forecast["probability"].to_numpy(dtype=np.float64) >= threshold).astype(np.int8)
        pooled[model_name] = forecast
        for cost_bps in config["per_side_cost_scenarios_bps"]:
            cost_bps = float(cost_bps)
            result = summarize_strategy(forecast, position, cost_bps, int(config["annualization_sessions"]))
            all_rows.append({"model": model_name, "per_side_cost_bps": cost_bps, **result})
            for year, group in forecast.groupby("fold_year", sort=True):
                group_positions = group.index.to_numpy()
                local_position = position[group_positions]
                fold_rows.append({
                    "model": model_name,
                    "fold_year": int(year),
                    "per_side_cost_bps": cost_bps,
                    **summarize_strategy(group.reset_index(drop=True), local_position,
                                         cost_bps, int(config["annualization_sessions"])),
                })

    always_long = pooled["selected_rc_sfg_seed_1729"].copy()
    always_long["probability"] = 1.0
    always_position = np.ones(len(always_long), dtype=np.int8)
    for cost_bps in config["per_side_cost_scenarios_bps"]:
        cost_bps = float(cost_bps)
        result = summarize_strategy(always_long, always_position, cost_bps, int(config["annualization_sessions"]))
        all_rows.append({"model": "always_long", "per_side_cost_bps": cost_bps, **result})
        for year, group in always_long.groupby("fold_year", sort=True):
            group_positions = group.index.to_numpy()
            fold_rows.append({
                "model": "always_long", "fold_year": int(year), "per_side_cost_bps": cost_bps,
                **summarize_strategy(group.reset_index(drop=True), always_position[group_positions],
                                     cost_bps, int(config["annualization_sessions"])),
            })

    pooled_rows = pd.DataFrame(all_rows)
    per_fold_rows = pd.DataFrame(fold_rows)
    breakeven_rows = []
    for model_name, forecast in {**pooled, "always_long": always_long}.items():
        position = np.ones(len(forecast), dtype=np.int8) if model_name == "always_long" else (
            forecast["probability"].to_numpy(dtype=np.float64) >= threshold
        ).astype(np.int8)
        breakeven_rows.append({
            "model": model_name,
            "development_breakeven_cost_per_side_bps": breakeven_side_cost_bps(forecast, position),
            "gross_total_return_positive": bool(final_wealth_at_cost(forecast, position, 0.0) > 1.0),
        })
    breakeven = pd.DataFrame(breakeven_rows)

    out = ROOT / "results/development"
    pooled_rows.to_csv(out / "economic_sensitivity_summary.csv", index=False, float_format="%.10g")
    per_fold_rows.to_csv(out / "economic_sensitivity_fold_metrics.csv", index=False, float_format="%.10g")
    breakeven.to_csv(out / "economic_sensitivity_breakeven.csv", index=False, float_format="%.10g")
    manifest = {
        "status": config["analysis_status"],
        "split": "development",
        "candidate_changed": False,
        "selection_or_tuning_use": False,
        "heldout_evaluation": False,
        "candidate_freeze_commit": freeze["frozen_commit"],
        "n_sessions": int(len(pooled["selected_rc_sfg_seed_1729"])),
        "forecast_origins_match": True,
        "threshold": threshold,
        "position_rule": config["position_rule"],
        "execution": config["execution"],
        "per_side_cost_scenarios_bps": config["per_side_cost_scenarios_bps"],
        "cost_definition": config["cost_definition"],
        "metric_definitions": {
            "annualized_return": "final compounded equity raised to 252/n minus one",
            "sharpe": "mean daily net return / sample SD, annualized by sqrt(252), zero risk-free rate",
            "max_drawdown": "minimum of compounded equity / running peak minus one, including initial equity 1",
            "turnover": "two times active-session fraction times 252, in notional multiples/year",
        },
        "not_modelled": ["market impact", "borrow costs", "taxes", "auction fill risk", "capital constraints"],
        "config_sha256": sha256(analysis_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "protocol_config_sha256": sha256(protocol_path),
        "candidate_freeze_manifest_sha256": sha256(freeze_path),
        "feature_table_sha256": sha256(feature_path),
        "baseline_predictions_sha256": sha256(ROOT / "results/development/baseline_predictions.csv"),
        "gate_signal_series_sha256": sha256(ROOT / "results/development/gate_diagnostic_signal_series.csv"),
        "files": {
            "pooled_summary": "results/development/economic_sensitivity_summary.csv",
            "fold_metrics": "results/development/economic_sensitivity_fold_metrics.csv",
            "breakeven": "results/development/economic_sensitivity_breakeven.csv",
        },
    }
    (out / "economic_sensitivity_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pooled_rows.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(breakeven.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Economic_Sensitivity_Status={manifest['status']}")


if __name__ == "__main__":
    main()
