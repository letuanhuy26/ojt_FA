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

from src.evaluation.walkforward import evaluate_models, load_yaml
from src.models.ours import FeatureGateLSTM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(pearsonr(x[valid], y[valid]).statistic)


def main() -> None:
    analysis_path = ROOT / "configs/stability_gate_analysis.yaml"
    base_path = ROOT / "configs/ours.yaml"
    protocol_path = ROOT / "configs/protocol.yaml"
    analysis = load_yaml(analysis_path)
    base = load_yaml(base_path)
    protocol = load_yaml(protocol_path)
    if analysis["split"] != "development":
        raise SystemExit("Stability-gate comparison is restricted to development.")
    if analysis["model"] != base["model"]:
        raise SystemExit("Analysis model does not match the frozen candidate.")
    seeds = [int(value) for value in analysis["model_seed_bases"]]

    active_capture: list[tuple[np.ndarray, np.ndarray]] = []
    original_forward = FeatureGateLSTM.forward

    def capture_forward(self, x, p):
        logits, gates = original_forward(self, x, p)
        if not self.training and self.mode == analysis["model"]:
            active_capture.append((p[:, -1].detach().cpu().numpy(), gates[:, -1].detach().cpu().numpy()))
        return logits, gates

    fold_seed_rows: list[dict] = []
    prediction_dates: list[list[str]] = []
    response_lags = [int(value) for value in analysis["response_lags_sessions"]]
    FeatureGateLSTM.forward = capture_forward
    try:
        for variant, overrides in analysis["variants"].items():
            for seed_base in seeds:
                active_capture.clear()
                config = dict(base)
                config.update({key: overrides[key] for key in ("lambda_sparsity", "lambda_stability")})
                config["seed"] = seed_base
                metrics, predictions = evaluate_models(
                    [analysis["model"]], config=config, protocol=protocol, split="development"
                )
                if len(active_capture) != len(metrics):
                    raise SystemExit(f"Incomplete endpoint-gate capture for {variant}, seed {seed_base}.")
                prediction_dates.append(predictions["target_date"].tolist())
                for (_, metric), (posterior, gates) in zip(metrics.sort_values("fold_year").iterrows(), active_capture):
                    p = posterior.astype(np.float64)
                    g = gates.astype(np.float64)
                    posterior_tv = np.full(len(p), np.nan)
                    gate_step = np.full(len(g), np.nan)
                    posterior_tv[1:] = 0.5 * np.abs(np.diff(p, axis=0)).sum(axis=1)
                    gate_step[1:] = np.abs(np.diff(g, axis=0)).mean(axis=1)
                    valid_tv = posterior_tv[np.isfinite(posterior_tv)]
                    cutoff = float(np.quantile(valid_tv, float(analysis["posterior_shift_event_quantile"])))
                    events = np.flatnonzero(np.isfinite(posterior_tv) & (posterior_tv >= cutoff))
                    row = {
                        "variant": variant,
                        "model_seed_base": seed_base,
                        "fold_seed": int(metric["seed"]),
                        "fold_year": int(metric["fold_year"]),
                        "n_origins": int(metric["n"]),
                        "mcc": float(metric["mcc"]),
                        "mean_endpoint_gate": float(g.mean()),
                        "mean_gate_step": float(np.nanmean(gate_step)),
                        "mean_posterior_tv": float(np.mean(valid_tv)),
                        "same_session_corr_tv_gate_step": correlation(posterior_tv, gate_step),
                        "lag1_corr_tv_gate_step": correlation(posterior_tv[:-1], gate_step[1:]),
                        "gate_step_after_p90_shift": float(np.nanmean(gate_step[events])),
                    }
                    for lag in response_lags:
                        event_positions = events + lag
                        event_positions = event_positions[event_positions < len(gate_step)]
                        row[f"gate_step_after_p90_shift_lag_{lag}"] = float(np.nanmean(gate_step[event_positions]))
                    fold_seed_rows.append(row)
    finally:
        FeatureGateLSTM.forward = original_forward

    if len(set(tuple(dates) for dates in prediction_dates)) != 1:
        raise SystemExit("Variants did not use identical forecast origins.")

    frame = pd.DataFrame(fold_seed_rows)
    expected = pd.read_csv(ROOT / "results/development/gate_ablation_fold_metrics.csv")
    for variant, overrides in analysis["variants"].items():
        ref_name = overrides["reference_ablation_variant"]
        ref = expected.loc[expected["variant"] == ref_name].set_index("fold_year")
        base_seed = int(base["seed"])
        replay = frame.loc[(frame["variant"] == variant) & (frame["model_seed_base"] == base_seed)].set_index("fold_year")
        for year, row in replay.iterrows():
            if abs(float(row["mcc"]) - float(ref.loc[year, "mcc"])) > 1e-8:
                raise SystemExit(f"Ablation-score replay mismatch for {variant}/{year}.")

    paired = frame.pivot(
        index=["model_seed_base", "fold_year"],
        columns="variant",
        values=[
            "mcc", "mean_gate_step", "same_session_corr_tv_gate_step", "lag1_corr_tv_gate_step",
            "gate_step_after_p90_shift", *[f"gate_step_after_p90_shift_lag_{lag}" for lag in response_lags],
        ],
    ).reset_index()
    paired.columns = ["_".join(str(part) for part in column if part) if isinstance(column, tuple) else column for column in paired.columns]
    paired["no_stability_minus_stability_mcc"] = paired["mcc_without_stability_penalty"] - paired["mcc_with_stability_penalty"]
    paired["no_stability_minus_stability_gate_step"] = paired["mean_gate_step_without_stability_penalty"] - paired["mean_gate_step_with_stability_penalty"]
    paired["no_stability_minus_stability_p90_response"] = paired["gate_step_after_p90_shift_without_stability_penalty"] - paired["gate_step_after_p90_shift_with_stability_penalty"]
    for lag in response_lags:
        column = f"gate_step_after_p90_shift_lag_{lag}"
        paired[f"no_stability_minus_stability_p90_response_lag_{lag}"] = (
            paired[f"{column}_without_stability_penalty"] - paired[f"{column}_with_stability_penalty"]
        )

    aggregations = dict(
        fold_seed_runs=("fold_year", "count"),
        mean_mcc=("mcc", "mean"),
        sd_fold_seed_mcc=("mcc", "std"),
        mean_gate_step=("mean_gate_step", "mean"),
        mean_endpoint_gate=("mean_endpoint_gate", "mean"),
        mean_same_session_correlation=("same_session_corr_tv_gate_step", "mean"),
        mean_lag1_correlation=("lag1_corr_tv_gate_step", "mean"),
        mean_gate_step_after_p90_shift=("gate_step_after_p90_shift", "mean"),
    )
    for lag in response_lags:
        aggregations[f"mean_gate_step_after_p90_shift_lag_{lag}"] = (
            f"gate_step_after_p90_shift_lag_{lag}", "mean"
        )
    variant_summary = frame.groupby("variant", sort=False).agg(**aggregations).reset_index()
    delta_row = {
        "matched_fold_seed_pairs": len(paired),
        "mean_no_stability_minus_stability_mcc": float(paired["no_stability_minus_stability_mcc"].mean()),
        "mean_no_stability_minus_stability_gate_step": float(paired["no_stability_minus_stability_gate_step"].mean()),
        "mean_no_stability_minus_stability_p90_response": float(paired["no_stability_minus_stability_p90_response"].mean()),
        "pairs_gate_step_higher_without_penalty": int((paired["no_stability_minus_stability_gate_step"] > 0).sum()),
        "pairs_gate_step_lower_without_penalty": int((paired["no_stability_minus_stability_gate_step"] < 0).sum()),
        "pairs_gate_step_equal_within_1e_8": int((paired["no_stability_minus_stability_gate_step"].abs() <= 1e-8).sum()),
        "inferential_test": "none",
    }
    for lag in response_lags:
        delta_row[f"mean_no_stability_minus_stability_p90_response_lag_{lag}"] = float(
            paired[f"no_stability_minus_stability_p90_response_lag_{lag}"].mean()
        )
    delta_summary = pd.DataFrame([delta_row])

    out = ROOT / "results/development"
    frame.to_csv(out / "stability_gate_fold_seed_metrics.csv", index=False, float_format="%.10g")
    paired.to_csv(out / "stability_gate_paired_deltas.csv", index=False, float_format="%.10g")
    variant_summary.to_csv(out / "stability_gate_variant_summary.csv", index=False, float_format="%.10g")
    delta_summary.to_csv(out / "stability_gate_delta_summary.csv", index=False, float_format="%.10g")
    manifest = {
        "status": analysis["analysis_status"],
        "split": "development",
        "model": analysis["model"],
        "variants": analysis["variants"],
        "model_seed_bases": seeds,
        "posterior_shift_event_quantile": float(analysis["posterior_shift_event_quantile"]),
        "response_lags_sessions": response_lags,
        "matched_fold_seed_pairs": int(len(paired)),
        "candidate_changed": False,
        "selection_or_tuning_use": False,
        "heldout_evaluation": False,
        "seed_1729_mcc_replays_match_penalty_ablation": True,
        "config_sha256": sha256(base_path),
        "protocol_sha256": sha256(protocol_path),
        "model_sha256": sha256(ROOT / "src/models/ours.py"),
        "evaluator_sha256": sha256(ROOT / "src/evaluation/walkforward.py"),
        "analysis_config_sha256": sha256(analysis_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "files": {
            "fold_seed_metrics": "results/development/stability_gate_fold_seed_metrics.csv",
            "paired_deltas": "results/development/stability_gate_paired_deltas.csv",
            "variant_summary": "results/development/stability_gate_variant_summary.csv",
            "delta_summary": "results/development/stability_gate_delta_summary.csv",
        },
    }
    (out / "stability_gate_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(variant_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(delta_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Stability_Gate_Status={manifest['status']}")


if __name__ == "__main__":
    main()
