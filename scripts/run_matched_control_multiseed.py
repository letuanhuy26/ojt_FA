from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.evaluation.walkforward import evaluate_models, load_yaml


RUN_ID = "matched-controls-20261004-02"
SEARCH_SUMMARY = ROOT / "results/development/matched_control_search_20261004/summary.csv"
SEED_BASES = [1729, 2026, 27182, 31415, 104729]
EXPECTED_FOLDS = [2015, 2016, 2017, 2018, 2019]
def main() -> None:
    print("[autoresearch] mode: classic")
    print("Goal: assess seed sensitivity of the best equally budgeted gate-mode configurations.")
    print("Metric: mean development MCC on the five locked folds; higher is better.")
    print("Verify: five development folds and 1,258 forecast origins per model-seed run.")
    print("Guard: development split only; no confirmation/final evaluation or model changes.")
    print(f"Seed bases: {SEED_BASES}")

    protocol = load_yaml(ROOT / "configs/protocol.yaml")
    if protocol["status"] != "frozen_for_confirmation_and_final":
        raise SystemExit("Refusing multi-seed replay: protocol is not marked frozen.")
    if protocol["evaluation"]["development_years"] != EXPECTED_FOLDS:
        raise SystemExit("Refusing multi-seed replay: locked development folds changed.")
    search = pd.read_csv(SEARCH_SUMMARY)
    best = search.loc[search.groupby("model")["mean_development_mcc"].idxmax()]
    modes = [
        "regime_dynamic_gate_lstm",
        "regime_as_feature_lstm",
        "dynamic_gate_lstm",
    ]
    if sorted(best["model"].tolist()) != sorted(modes):
        raise SystemExit("Matched search does not contain exactly the three planned model modes.")

    run_dir = ROOT / "autoresearch" / RUN_ID
    out_dir = ROOT / "results/development/matched_control_multiseed_20261004"
    run_dir.mkdir(parents=True, exist_ok=False)
    out_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = run_dir / "results.tsv"
    fold_path = out_dir / "fold_seed_metrics.csv"
    seed_path = out_dir / "seed_summary.csv"
    mode_path = out_dir / "mode_summary.csv"
    prediction_path = out_dir / "predictions.csv"
    run_path = out_dir / "run.json"

    best_configs = {}
    for row in best.itertuples(index=False):
        best_configs[row.model] = {
            "candidate": row.candidate,
            "mean_mcc_at_seed_1729": float(row.mean_development_mcc),
            "config": json.loads(row.configuration),
        }

    all_metrics = []
    all_predictions = []
    ledger_fields = [
        "model", "best_candidate", "seed_base", "timestamp_utc", "mean_development_mcc",
        "fold_mccs", "n_forecasts", "verify_wall_seconds", "verdict",
    ]
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ledger_fields, delimiter="\t")
        writer.writeheader()
        handle.flush()
        for model in modes:
            chosen = best_configs[model]
            for seed_base in SEED_BASES:
                config = {**chosen["config"], "model": model, "seed": seed_base}
                started = time.perf_counter()
                metrics, predictions = evaluate_models(
                    [model], config=config, protocol=protocol, split="development"
                )
                elapsed = time.perf_counter() - started
                years = [int(value) for value in metrics["fold_year"].tolist()]
                n_forecasts = int(metrics["n"].sum())
                if years != EXPECTED_FOLDS or len(metrics) != 5 or n_forecasts != 1258:
                    raise SystemExit(
                        f"Verify failed for {model}/seed={seed_base}: folds={years}, n={n_forecasts}"
                    )
                fold_mccs = metrics["mcc"].astype(float).tolist()
                if any(not pd.notna(value) for value in fold_mccs):
                    raise SystemExit(f"Verify failed for {model}/seed={seed_base}: non-finite MCC")
                mean_mcc = float(sum(fold_mccs) / len(fold_mccs))
                if seed_base == 1729 and abs(mean_mcc - chosen["mean_mcc_at_seed_1729"]) > 1e-8:
                    raise SystemExit(
                        f"Seed-1729 replay mismatch for {model}: {mean_mcc} vs "
                        f"{chosen['mean_mcc_at_seed_1729']}"
                    )
                for row in metrics.to_dict(orient="records"):
                    all_metrics.append({
                        **row,
                        "seed_base": seed_base,
                        "best_candidate": chosen["candidate"],
                        "configuration": json.dumps(config, sort_keys=True, separators=(",", ":")),
                        "verify_wall_seconds": elapsed,
                    })
                if not predictions.empty:
                    predictions = predictions.copy()
                    predictions["seed_base"] = seed_base
                    predictions["best_candidate"] = chosen["candidate"]
                    all_predictions.extend(predictions.to_dict(orient="records"))
                writer.writerow({
                    "model": model,
                    "best_candidate": chosen["candidate"],
                    "seed_base": seed_base,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "mean_development_mcc": f"{mean_mcc:.10f}",
                    "fold_mccs": "|".join(f"{value:.10f}" for value in fold_mccs),
                    "n_forecasts": n_forecasts,
                    "verify_wall_seconds": f"{elapsed:.3f}",
                    "verdict": "REPLAY",
                })
                handle.flush()
                print(f"{model}\tseed={seed_base}\tMCC={mean_mcc:.6f}\t{elapsed:.1f}s", flush=True)

    fold_metrics = pd.DataFrame(all_metrics)
    fold_metrics.to_csv(fold_path, index=False, float_format="%.10g")
    pd.DataFrame(all_predictions).to_csv(prediction_path, index=False, float_format="%.10g")
    seed_summary = fold_metrics.groupby(["model", "seed_base"], sort=False).agg(
        mean_mcc=("mcc", "mean"),
        sd_fold_mcc=("mcc", "std"),
        positive_folds=("mcc", lambda values: int((values > 0).sum())),
        forecasts=("n", "sum"),
    ).reset_index()
    seed_summary.to_csv(seed_path, index=False, float_format="%.10g")
    mode_summary = fold_metrics.groupby("model", sort=False).agg(
        fold_seed_runs=("mcc", "count"),
        mean_mcc=("mcc", "mean"),
        sd_fold_seed_mcc=("mcc", "std"),
        positive_fold_seed_runs=("mcc", lambda values: int((values > 0).sum())),
    ).reset_index()
    seed_sd = seed_summary.groupby("model")["mean_mcc"].std()
    mode_summary["mean_seed_level_mcc_sd"] = mode_summary["model"].map(seed_sd)
    best_candidate_map = {model: best_configs[model]["candidate"] for model in modes}
    mode_summary["best_candidate"] = mode_summary["model"].map(best_candidate_map)
    mode_summary.to_csv(mode_path, index=False, float_format="%.10g")

    metadata = {
        "run_id": RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "development_only",
        "heldout_accessed": False,
        "fold_years": EXPECTED_FOLDS,
        "seed_bases": SEED_BASES,
        "models": modes,
        "selected_configs_from": "results/development/matched_control_search_20261004/summary.csv",
        "seeds_per_mode": len(SEED_BASES),
        "fold_seed_runs_per_mode": len(SEED_BASES) * len(EXPECTED_FOLDS),
        "interpretation": "post-hoc seed robustness replay of configurations selected on the same development folds; descriptive only",
        "outputs": {
            "fold_seed_metrics": str(fold_path.relative_to(ROOT)),
            "seed_summary": str(seed_path.relative_to(ROOT)),
            "mode_summary": str(mode_path.relative_to(ROOT)),
            "predictions": str(prediction_path.relative_to(ROOT)),
        },
    }
    run_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "# Matched gate-mode seed robustness replay",
        "",
        "For each model mode, the best configuration was selected by the seed-1729 matched sweep, then replayed over five seed bases and the same five development folds. The replay is post hoc on the model-selection data, uses no held-out period, and is descriptive rather than confirmatory.",
        "",
        "| Model | Selected config | Mean MCC over 25 fold-seed runs | SD across fold-seed runs | Positive runs | SD of five seed-level means |",
        "|---|---|---:|---:|---:|---:|",
    ]
    mode_summary_by_name = mode_summary.set_index("model")
    for model in modes:
        row = mode_summary_by_name.loc[model]
        lines.append(
            f"| {model} | {row['best_candidate']} | {row['mean_mcc']:.5f} | "
            f"{row['sd_fold_seed_mcc']:.5f} | {int(row['positive_fold_seed_runs'])}/25 | "
            f"{row['mean_seed_level_mcc_sd']:.5f} |"
        )
    lines.extend([
        "",
        "Per-seed fold means and the complete replay ledger are in `seed_summary.csv` and `results.tsv`. Raw fold/forecast outputs and configuration hashes are in `results/development/matched_control_multiseed_20261004/`.",
    ])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nSaved development-only seed robustness results:")
    print(mode_summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))


if __name__ == "__main__":
    main()
