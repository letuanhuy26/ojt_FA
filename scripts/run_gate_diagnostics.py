from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import FEATURE_COLUMNS
from src.evaluation.walkforward import evaluate_models, load_examples, load_yaml
from src.models.ours import FeatureGateLSTM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(pearsonr(left[valid], right[valid]).statistic)


def main() -> None:
    analysis_path = ROOT / "configs/gate_diagnostics.yaml"
    config_path = ROOT / "configs/ours.yaml"
    protocol_path = ROOT / "configs/protocol.yaml"
    analysis = load_yaml(analysis_path)
    config = load_yaml(config_path)
    protocol = load_yaml(protocol_path)
    if analysis["split"] != "development":
        raise SystemExit("Gate diagnostics are restricted to development.")
    if config["model"] != analysis["model"]:
        raise SystemExit("Diagnostic model differs from the frozen candidate.")
    seed_bases = [int(value) for value in analysis["model_seed_bases"]]
    if len(seed_bases) != len(set(seed_bases)) or int(config["seed"]) not in seed_bases:
        raise SystemExit("Seed bases must be unique and include the frozen candidate replay seed.")

    active_capture: list[tuple[np.ndarray, np.ndarray]] = []
    original_forward = FeatureGateLSTM.forward

    def capture_forward(self, x, p):
        logits, gates = original_forward(self, x, p)
        if not self.training and self.mode == analysis["model"]:
            active_capture.append((p[:, -1].detach().cpu().numpy(), gates[:, -1].detach().cpu().numpy()))
        return logits, gates

    metric_runs: list[pd.DataFrame] = []
    prediction_runs: list[pd.DataFrame] = []
    captures_by_seed: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    FeatureGateLSTM.forward = capture_forward
    try:
        for seed_base in seed_bases:
            active_capture.clear()
            seed_config = dict(config)
            seed_config["seed"] = seed_base
            metrics, predictions = evaluate_models(
                [analysis["model"]], config=seed_config, protocol=protocol, split="development"
            )
            if len(active_capture) != len(metrics):
                raise SystemExit(
                    f"Expected {len(metrics)} captured validation passes for seed {seed_base}; "
                    f"got {len(active_capture)}."
                )
            metrics["model_seed_base"] = seed_base
            predictions["model_seed_base"] = seed_base
            metric_runs.append(metrics)
            prediction_runs.append(predictions)
            captures_by_seed[seed_base] = list(active_capture)
    finally:
        FeatureGateLSTM.forward = original_forward

    all_metrics = pd.concat(metric_runs, ignore_index=True)
    all_predictions = pd.concat(prediction_runs, ignore_index=True)
    expected = pd.read_csv(ROOT / "results/development/gate_ablation_fold_metrics.csv")
    expected = expected.loc[expected["variant"] == "selected_candidate"].set_index("fold_year")
    frozen_seed_metrics = all_metrics.loc[
        all_metrics["model_seed_base"] == int(config["seed"])
    ].set_index("fold_year")
    tolerance = float(analysis["metric_replay_tolerance"])
    for year, row in frozen_seed_metrics.iterrows():
        target = float(expected.loc[year, "mcc"])
        if abs(float(row["mcc"]) - target) > tolerance:
            raise SystemExit(f"Frozen-seed replay mismatch in {year}: {row['mcc']:.12f} vs {target:.12f}.")

    examples = load_examples()
    fold_rows: list[dict] = []
    feature_rows: list[dict] = []
    signal_frames: list[pd.DataFrame] = []
    q = float(analysis["posterior_shift_event_quantile"])
    response_lags = [int(value) for value in analysis["response_lags_sessions"]]

    for seed_base in seed_bases:
        observed_seed = all_metrics.loc[all_metrics["model_seed_base"] == seed_base].set_index("fold_year")
        predictions_seed = all_predictions.loc[all_predictions["model_seed_base"] == seed_base]
        for (year, model_row), (posterior, gates) in zip(
            observed_seed.sort_index().iterrows(), captures_by_seed[seed_base]
        ):
            preds = predictions_seed.loc[predictions_seed["fold_year"] == year]
            preds = preds.sort_values("target_date").reset_index(drop=True)
            preds["target_date"] = pd.to_datetime(preds["target_date"])
            date_map = examples[["date", "target_date"]].drop_duplicates("target_date")
            aligned = preds.merge(date_map, on="target_date", how="left", validate="one_to_one")
            if aligned["date"].isna().any() or len(aligned) != len(posterior) or len(aligned) != len(gates):
                raise SystemExit(f"Signal dates, posterior, gates, and predictions do not align in {year}, seed {seed_base}.")

            frame = pd.DataFrame({
                "model_seed_base": seed_base,
                "fold_seed": int(model_row["seed"]),
                "fold_year": int(year),
                "signal_date": aligned["date"].dt.strftime("%Y-%m-%d"),
                "target_date": aligned["target_date"],
                "y_true": aligned["y_true"],
                "probability": aligned["probability"],
            })
            for index in range(posterior.shape[1]):
                frame[f"filtered_state_probability_{index}"] = posterior[:, index]
            for index, name in enumerate(FEATURE_COLUMNS):
                frame[f"gate_{name}"] = gates[:, index]

            p = posterior.astype(np.float64)
            g = gates.astype(np.float64)
            posterior_tv = np.full(len(p), np.nan)
            gate_step = np.full(len(g), np.nan)
            posterior_tv[1:] = 0.5 * np.abs(np.diff(p, axis=0)).sum(axis=1)
            gate_step[1:] = np.abs(np.diff(g, axis=0)).mean(axis=1)
            normalized_entropy = -(
                np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0))
            ).sum(axis=1) / np.log(p.shape[1])
            frame["posterior_total_variation"] = posterior_tv
            frame["mean_absolute_gate_step"] = gate_step
            frame["filtered_state_argmax"] = posterior.argmax(axis=1)
            frame["argmax_state_changed"] = np.r_[
                False, posterior.argmax(axis=1)[1:] != posterior.argmax(axis=1)[:-1]
            ]
            signal_frames.append(frame)

            valid_tv = posterior_tv[np.isfinite(posterior_tv)]
            shift_cutoff = float(np.quantile(valid_tv, q))
            events = np.flatnonzero(np.isfinite(posterior_tv) & (posterior_tv >= shift_cutoff))
            row = {
                "model_seed_base": seed_base,
                "fold_seed": int(model_row["seed"]),
                "fold_year": int(year),
                "n_origins": int(len(frame)),
                "signal_start": str(frame["signal_date"].iloc[0]),
                "signal_end": str(frame["signal_date"].iloc[-1]),
                "replayed_mcc": float(model_row["mcc"]),
                "mean_posterior_tv": float(np.mean(valid_tv)),
                "p90_posterior_tv": shift_cutoff,
                "mean_max_state_probability": float(p.max(axis=1).mean()),
                "mean_normalized_state_entropy": float(normalized_entropy.mean()),
                "argmax_state_switches": int(frame["argmax_state_changed"].sum()),
                "mean_endpoint_gate": float(g.mean()),
                "mean_gate_step": float(np.nanmean(gate_step)),
                "gate_step_posterior_tv_pearson_r": finite_correlation(posterior_tv, gate_step),
                "high_shift_events": int(len(events)),
            }
            for state_index in range(p.shape[1]):
                row[f"mean_filtered_state_probability_{state_index}"] = float(p[:, state_index].mean())
            for lag in response_lags:
                stop = len(posterior_tv) - lag
                row[f"gate_step_corr_tv_lag_{lag}"] = finite_correlation(
                    posterior_tv[:stop], gate_step[lag:]
                )
                event_positions = events + lag
                event_positions = event_positions[event_positions < len(gate_step)]
                row[f"mean_gate_step_after_p90_shift_lag_{lag}"] = (
                    float(np.nanmean(gate_step[event_positions])) if len(event_positions) else float("nan")
                )
            fold_rows.append(row)

            per_feature_step = np.abs(np.diff(g, axis=0))
            for index, name in enumerate(FEATURE_COLUMNS):
                feature_rows.append({
                    "model_seed_base": seed_base,
                    "fold_seed": int(model_row["seed"]),
                    "fold_year": int(year),
                    "feature": name,
                    "mean_endpoint_gate": float(g[:, index].mean()),
                    "sd_endpoint_gate": float(g[:, index].std(ddof=1)),
                    "fraction_gate_below_0_1": float((g[:, index] < 0.1).mean()),
                    "fraction_gate_above_0_9": float((g[:, index] > 0.9).mean()),
                    "mean_absolute_gate_step": float(per_feature_step[:, index].mean()),
                })

    fold_summary = pd.DataFrame(fold_rows)
    feature_summary = pd.DataFrame(feature_rows)
    signals = pd.concat(signal_frames, ignore_index=True)
    seed_summary = fold_summary.groupby("fold_year", sort=True).agg(
        seeds=("model_seed_base", "nunique"),
        mean_mcc=("replayed_mcc", "mean"),
        sd_seed_mcc=("replayed_mcc", "std"),
        min_mcc=("replayed_mcc", "min"),
        max_mcc=("replayed_mcc", "max"),
        mean_gate_step=("mean_gate_step", "mean"),
        sd_seed_gate_step=("mean_gate_step", "std"),
        mean_endpoint_gate=("mean_endpoint_gate", "mean"),
        sd_seed_endpoint_gate=("mean_endpoint_gate", "std"),
    ).reset_index()
    polarity = fold_summary.groupby("fold_year", sort=True).agg(
        all_seed_mcc_positive=("replayed_mcc", lambda values: bool((values > 0).all())),
        all_seed_mcc_negative=("replayed_mcc", lambda values: bool((values < 0).all())),
    ).reset_index()
    gate_columns = [f"gate_{name}" for name in FEATURE_COLUMNS]
    gate_variability_rows = []
    for year, group in signals.groupby("fold_year", sort=True):
        seed_gate_arrays = [
            part.sort_values("signal_date")[gate_columns].to_numpy(dtype=np.float64)
            for _, part in group.groupby("model_seed_base", sort=True)
        ]
        gate_sd = np.stack(seed_gate_arrays, axis=0).std(axis=0, ddof=1)
        gate_variability_rows.append({
            "fold_year": int(year),
            "seeds": int(len(seed_gate_arrays)),
            "mean_between_seed_gate_sd": float(gate_sd.mean()),
            "median_between_seed_gate_sd": float(np.median(gate_sd)),
            "fraction_date_feature_seed_sd_gt_0_1": float((gate_sd > 0.1).mean()),
            "max_between_seed_gate_sd": float(gate_sd.max()),
        })
    gate_variability = pd.DataFrame(gate_variability_rows)
    seed_summary = seed_summary.merge(polarity, on="fold_year", validate="one_to_one")
    seed_summary = seed_summary.merge(gate_variability, on=["fold_year", "seeds"], validate="one_to_one")
    seed_base_summary = fold_summary.groupby("model_seed_base", sort=True).agg(
        folds=("fold_year", "nunique"),
        mean_mcc=("replayed_mcc", "mean"),
        sd_fold_mcc=("replayed_mcc", "std"),
        min_fold_mcc=("replayed_mcc", "min"),
        max_fold_mcc=("replayed_mcc", "max"),
        positive_folds=("replayed_mcc", lambda values: int((values > 0).sum())),
        negative_folds=("replayed_mcc", lambda values: int((values < 0).sum())),
    ).reset_index()

    out = ROOT / "results/development"
    out.mkdir(parents=True, exist_ok=True)
    fold_summary.to_csv(out / "gate_diagnostic_fold_summary.csv", index=False, float_format="%.10g")
    feature_summary.to_csv(out / "gate_diagnostic_feature_summary.csv", index=False, float_format="%.10g")
    seed_summary.to_csv(out / "gate_diagnostic_seed_summary.csv", index=False, float_format="%.10g")
    seed_base_summary.to_csv(out / "gate_diagnostic_seed_base_summary.csv", index=False, float_format="%.10g")
    signals.to_csv(out / "gate_diagnostic_signal_series.csv", index=False, float_format="%.10g")
    overall = {
        "n_seed_bases": len(seed_bases),
        "n_fold_seed_runs": int(len(fold_summary)),
        "mean_fold_seed_mcc": float(fold_summary["replayed_mcc"].mean()),
        "sd_fold_seed_mcc": float(fold_summary["replayed_mcc"].std(ddof=1)),
        "min_fold_seed_mcc": float(fold_summary["replayed_mcc"].min()),
        "max_fold_seed_mcc": float(fold_summary["replayed_mcc"].max()),
        "fraction_fold_seed_runs_with_positive_mcc": float((fold_summary["replayed_mcc"] > 0).mean()),
        "mean_between_seed_gate_sd": float(seed_summary["mean_between_seed_gate_sd"].mean()),
    }
    manifest = {
        "status": analysis["analysis_status"],
        "split": "development",
        "model": analysis["model"],
        "candidate_changed": False,
        "selection_or_tuning_use": False,
        "model_seed_bases": seed_bases,
        "seed_1729_replay_matches_ablation": True,
        "overall_seed_summary": overall,
        "captured_gate": "last time step of each 60-session input window, one value per forecast origin",
        "posterior_shift": "one-half L1 distance between adjacent causal filtered posteriors",
        "high_shift_event_quantile": q,
        "response_lags_sessions": response_lags,
        "config_sha256": sha256(config_path),
        "protocol_sha256": sha256(protocol_path),
        "model_sha256": sha256(ROOT / "src/models/ours.py"),
        "evaluator_sha256": sha256(ROOT / "src/evaluation/walkforward.py"),
        "diagnostic_config_sha256": sha256(analysis_path),
        "diagnostic_script_sha256": sha256(Path(__file__)),
        "files": {
            "fold_summary": "results/development/gate_diagnostic_fold_summary.csv",
            "feature_summary": "results/development/gate_diagnostic_feature_summary.csv",
            "seed_summary": "results/development/gate_diagnostic_seed_summary.csv",
            "seed_base_summary": "results/development/gate_diagnostic_seed_base_summary.csv",
            "signal_series": "results/development/gate_diagnostic_signal_series.csv",
        },
    }
    (out / "gate_diagnostic_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(seed_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(seed_base_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Overall_Seed_Summary={json.dumps(overall, sort_keys=True)}")
    print(f"Gate_Diagnostic_Status={manifest['status']}")


if __name__ == "__main__":
    main()
