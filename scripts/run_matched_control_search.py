from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.walkforward import evaluate_models, load_yaml


RUN_ID = "matched-controls-20261004-01"
MODELS = [
    "regime_dynamic_gate_lstm",
    "dynamic_gate_lstm",
    "regime_as_feature_lstm",
]
ITERATIONS = [
    ("dropout_01", {"dropout": 0.1}),
    ("window_10", {"window": 10}),
    ("window_60", {"window": 60}),
    ("epochs_10", {"epochs": 10}),
    ("learning_rate_0005", {"learning_rate": 0.0005}),
    ("weight_decay_001", {"weight_decay": 0.001}),
    ("hidden_size_16", {"hidden_size": 16}),
    ("hidden_size_64", {"hidden_size": 64}),
    ("batch_size_128", {"batch_size": 128}),
    ("batch_size_512", {"batch_size": 512}),
]
TSV_FIELDS = [
    "model", "iteration", "timestamp_utc", "candidate", "configuration",
    "mean_development_mcc", "fold_mccs", "verify_wall_seconds", "verdict",
]


def baseline_config() -> dict:
    config = load_yaml(ROOT / "configs/ours.yaml")
    # Reconstruct the original pre-search RC-SFG configuration from the
    # existing model-campaign ledger so all three modes share its starting point.
    config.update({"window": 20, "weight_decay": 0.0001})
    return config


def run() -> None:
    print("[autoresearch] mode: classic")
    print("Goal: compare equal-budget development searches for RC-SFG and two close controls.")
    print("Metric: unweighted mean MCC over the five locked development folds; higher is better.")
    print("Verify: evaluate_models(mode, config, protocol, split='development')")
    print("Guard: frozen protocol; development split only; no confirmation/final calls.")
    print("Iterations: 10 candidate configurations per model, plus each fixed baseline.")

    protocol = load_yaml(ROOT / "configs/protocol.yaml")
    if protocol["status"] != "frozen_for_confirmation_and_final":
        raise SystemExit("Refusing matched search: protocol is not marked frozen.")
    expected_folds = [2015, 2016, 2017, 2018, 2019]
    if protocol["evaluation"]["development_years"] != expected_folds:
        raise SystemExit("Refusing matched search: development folds differ from the locked pilot.")

    run_dir = ROOT / "autoresearch" / RUN_ID
    out_dir = ROOT / "results/development/matched_control_search_20261004"
    run_dir.mkdir(parents=True, exist_ok=False)
    out_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = run_dir / "results.tsv"
    metrics_path = out_dir / "fold_metrics.csv"
    predictions_path = out_dir / "predictions.csv"
    summary_path = out_dir / "summary.csv"
    run_path = out_dir / "run.json"

    base = baseline_config()
    base["seed"] = int(protocol["evaluation"]["random_seed"])
    baseline_name = "baseline"
    candidates = [(baseline_name, {})] + ITERATIONS
    best_by_model: dict[str, float] = {model: float("-inf") for model in MODELS}
    summary_rows = []
    all_fold_rows = []
    all_prediction_rows = []

    with ledger_path.open("w", newline="", encoding="utf-8") as ledger_file:
        writer = csv.DictWriter(ledger_file, fieldnames=TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        ledger_file.flush()
        for model in MODELS:
            for iteration, (candidate_name, overrides) in enumerate(candidates):
                config = {**base, **overrides, "model": model}
                started = time.perf_counter()
                metrics, predictions = evaluate_models(
                    [model], config=config, protocol=protocol, split="development"
                )
                elapsed = time.perf_counter() - started
                fold_years = [int(value) for value in metrics["fold_year"].tolist()]
                if fold_years != expected_folds:
                    raise SystemExit(f"Verify failed for {model}/{candidate_name}: folds={fold_years}")
                if int(metrics["n"].sum()) != 1258:
                    raise SystemExit(f"Verify failed for {model}/{candidate_name}: forecast count={int(metrics['n'].sum())}")
                fold_mccs = metrics["mcc"].astype(float).tolist()
                mean_mcc = sum(fold_mccs) / len(fold_mccs)
                if len(fold_mccs) != 5 or any(value != value for value in fold_mccs):
                    raise SystemExit(f"Verify failed for {model}/{candidate_name}: missing/nonfinite fold MCC")
                verdict = "BASELINE" if iteration == 0 else ("NEW_BEST" if mean_mcc > best_by_model[model] else "NO_IMPROVEMENT")
                best_by_model[model] = max(best_by_model[model], mean_mcc)
                config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
                summary_rows.append({
                    "model": model,
                    "iteration": iteration,
                    "candidate": candidate_name,
                    "configuration": config_json,
                    "mean_development_mcc": mean_mcc,
                    "sd_fold_mcc": float(metrics["mcc"].std(ddof=1)),
                    "mean_balanced_accuracy": float(metrics["balanced_accuracy"].mean()),
                    "mean_auroc": float(metrics["auroc"].mean()),
                    "mean_brier": float(metrics["brier"].mean()),
                    "mean_log_loss": float(metrics["log_loss"].mean()),
                    "verify_wall_seconds": elapsed,
                    "verdict": verdict,
                })
                for row in metrics.to_dict(orient="records"):
                    all_fold_rows.append({
                        **row,
                        "candidate": candidate_name,
                        "iteration": iteration,
                        "configuration": config_json,
                        "verify_wall_seconds": elapsed,
                    })
                if not predictions.empty:
                    prediction_rows = predictions.copy()
                    prediction_rows["candidate"] = candidate_name
                    prediction_rows["iteration"] = iteration
                    all_prediction_rows.extend(prediction_rows.to_dict(orient="records"))
                writer.writerow({
                    "model": model,
                    "iteration": iteration,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "candidate": candidate_name,
                    "configuration": config_json,
                    "mean_development_mcc": f"{mean_mcc:.10f}",
                    "fold_mccs": "|".join(f"{value:.10f}" for value in fold_mccs),
                    "verify_wall_seconds": f"{elapsed:.3f}",
                    "verdict": verdict,
                })
                ledger_file.flush()
                print(f"{model}	{iteration:02d}	{candidate_name}	MCC={mean_mcc:.6f}	{verdict}	{elapsed:.1f}s", flush=True)

    import pandas as pd

    pd.DataFrame(all_fold_rows).to_csv(metrics_path, index=False, float_format="%.10g")
    pd.DataFrame(all_prediction_rows).to_csv(predictions_path, index=False, float_format="%.10g")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False, float_format="%.10g")
    best_rows = summary.loc[summary.groupby("model")["mean_development_mcc"].idxmax()].sort_values("model")
    fold_metrics_frame = pd.DataFrame(all_fold_rows)
    metadata = {
        "run_id": RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "unweighted mean development MCC at threshold 0.5",
        "split": "development_only",
        "fold_years": expected_folds,
        "models": MODELS,
        "candidate_configurations_per_model": len(ITERATIONS),
        "fixed_baseline_per_model": True,
        "starting_config": base,
        "search_space": [{"candidate": name, "overrides": values} for name, values in ITERATIONS],
        "results": [
            {"model": row.model, "candidate": row.candidate,
             "mean_development_mcc": float(row.mean_development_mcc),
             "sd_fold_mcc": float(row.sd_fold_mcc),
             "fold_mccs": fold_metrics_frame[
                 (fold_metrics_frame["model"] == row.model) &
                 (fold_metrics_frame["candidate"] == row.candidate)
             ].sort_values("fold_year")["mcc"].astype(float).tolist()}
            for row in best_rows.itertuples(index=False)
        ],
        "heldout_accessed": False,
        "interpretation": "post-hoc equal-budget development comparison; not used to change the frozen candidate and not confirmatory",
        "outputs": {
            "fold_metrics": str(metrics_path.relative_to(ROOT)),
            "predictions": str(predictions_path.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
        },
    }
    run_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    best_table = best_rows[["model", "candidate", "mean_development_mcc", "sd_fold_mcc"]]
    table_lines = [
        "| Model | Best candidate | Mean development MCC | SD across folds |",
        "|---|---|---:|---:|",
    ]
    for row in best_table.itertuples(index=False):
        table_lines.append(
            f"| {row.model} | {row.candidate} | {row.mean_development_mcc:.5f} | {row.sd_fold_mcc:.5f} |"
        )
    (run_dir / "summary.md").write_text(
        "# Matched development control search\n\n"
        "All models received the same fixed baseline and ten one-parameter-at-a-time candidate configurations on the same five development folds, with one deterministic seed per fold. The search is post hoc, uses the model-selection period, and does not reopen confirmation or final. The frozen RC-SFG configuration was not changed.\n\n"
        + "\n".join(table_lines)
        + "\n\nComplete iteration ledger: `results.tsv`. Fold metrics, predictions, summary, and run metadata are under `results/development/matched_control_search_20261004/`.\n",
        encoding="utf-8",
    )
    print("\nBest configuration per model:")
    print(best_table.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Saved development-only outputs in {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
