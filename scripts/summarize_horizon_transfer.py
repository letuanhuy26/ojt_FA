from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/horizon_transfer_20261004"


def main() -> None:
    metrics = pd.read_csv(OUT / "fold_metrics.csv")
    bootstrap = pd.read_csv(OUT / "heldout_mcc_bootstrap.csv")
    expected_models = {
        "regime_dynamic_gate_lstm",
        "dynamic_gate_lstm",
        "regime_as_feature_lstm",
        "random_forest",
    }
    if set(metrics["model"].unique()) != expected_models:
        raise SystemExit("Unexpected model set in horizon-transfer metrics.")
    if set(metrics["horizon_sessions"].unique()) != {5, 20}:
        raise SystemExit("Unexpected horizons in horizon-transfer metrics.")
    if set(metrics["split"].unique()) != {"development", "confirmation", "final"}:
        raise SystemExit("Unexpected split set in horizon-transfer metrics.")

    development = metrics[metrics["split"] == "development"]
    if (development.groupby(["horizon_sessions", "model"]).size() != 5).any():
        raise SystemExit("Every development model/horizon must have exactly five fold scores.")
    dev_summary = development.groupby(["horizon_sessions", "model"], sort=False).agg(
        folds=("fold_year", "count"),
        mean_mcc=("mcc", "mean"),
        sd_fold_mcc=("mcc", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_auroc=("auroc", "mean"),
        mean_brier=("brier", "mean"),
        mean_log_loss=("log_loss", "mean"),
        mean_positive_rate=("positive_rate", "mean"),
        mean_predicted_positive_rate=("predicted_positive_rate", "mean"),
        positive_folds=("mcc", lambda values: int((values > 0).sum())),
        n_forecasts=("n", "sum"),
    ).reset_index()
    dev_summary.to_csv(OUT / "development_summary.csv", index=False, float_format="%.10g")

    heldout = metrics[metrics["split"].isin(["confirmation", "final"])][
        ["horizon_sessions", "split", "model", "n", "positive_rate", "predicted_positive_rate",
         "mcc", "balanced_accuracy", "accuracy", "auroc", "brier", "log_loss"]
    ].copy()
    heldout.to_csv(OUT / "heldout_metrics.csv", index=False, float_format="%.10g")

    order = [
        "regime_dynamic_gate_lstm",
        "dynamic_gate_lstm",
        "regime_as_feature_lstm",
        "random_forest",
    ]
    dev_labels = {
        "regime_dynamic_gate_lstm": "RC-SFG",
        "dynamic_gate_lstm": "Dynamic gate, no regime",
        "regime_as_feature_lstm": "Regime as feature",
        "random_forest": "Random Forest",
    }
    lines = [
        "# Cross-horizon transfer results (exploratory)",
        "",
        "The h=1 frozen configuration was transferred unchanged to five-session and twenty-session open-to-close targets. No horizon-specific tuning was performed. All results are post-hoc because the same market and calendar blocks have already been examined for h=1.",
        "",
        "## Development performance",
        "",
        "| Horizon | Model | Mean MCC | SD across five folds | Positive folds | Mean AUROC | Mean predicted-positive rate |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for horizon in [5, 20]:
        subset = dev_summary[dev_summary["horizon_sessions"] == horizon].set_index("model")
        for model in order:
            row = subset.loc[model]
            lines.append(
                f"| {horizon} | {dev_labels[model]} | {row['mean_mcc']:.5f} | "
                f"{row['sd_fold_mcc']:.5f} | {int(row['positive_folds'])}/5 | "
                f"{row['mean_auroc']:.5f} | {row['mean_predicted_positive_rate']:.3f} |"
            )
    lines.extend([
        "",
        "## Confirmation and final MCC",
        "",
        "| Horizon | Period | n | Model | MCC | Balanced accuracy | AUROC | Predicted positive |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ])
    heldout_order = heldout.copy()
    heldout_order["model_order"] = heldout_order["model"].map({name: i for i, name in enumerate(order)})
    heldout_order["split_order"] = heldout_order["split"].map({"confirmation": 0, "final": 1})
    heldout_order = heldout_order.sort_values(["horizon_sessions", "split_order", "model_order"])
    for row in heldout_order.itertuples(index=False):
        label = "Confirmation 2020–2021" if row.split == "confirmation" else "Final 2022–2025"
        lines.append(
            f"| {row.horizon_sessions} | {label} | {int(row.n)} | {dev_labels[row.model]} | "
            f"{row.mcc:.5f} | {row.balanced_accuracy:.5f} | {row.auroc:.5f} | "
            f"{row.predicted_positive_rate:.3f} |"
        )
    lines.extend([
        "",
        "## Paired RC-SFG minus Random Forest MCC",
        "",
        "| Horizon | Period | Δ MCC | 95% circular block-bootstrap interval |",
        "|---:|---|---:|---:|",
    ])
    for row in bootstrap.itertuples(index=False):
        label = "Confirmation 2020–2021" if row.split == "confirmation" else "Final 2022–2025"
        lines.append(
            f"| {row.horizon_sessions} | {label} | {row.point_delta_mcc:.5f} | "
            f"[{row.ci95_percentile_low:.5f}, {row.ci95_percentile_high:.5f}] |"
        )
    lines.extend([
        "",
        "Intervals use 10,000 paired circular blocks of 20 forecast origins, seed 1729; no p-values are computed. The h-session targets overlap because one signal is evaluated at each session. Thus this is a predictive transfer diagnostic, not a portfolio backtest or an economic evaluation. All held-out results are exploratory and do not replace a prospectively frozen multi-horizon protocol.",
        "",
        "Machine-readable outputs are `fold_metrics.csv`, `development_summary.csv`, `heldout_metrics.csv`, `heldout_mcc_bootstrap.csv`, and `run.json`. Raw predictions and derived horizon labels remain local.",
    ])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_path = OUT / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["outputs"]["development_summary"] = "results/horizon_transfer_20261004/development_summary.csv"
    run["outputs"]["heldout_metrics"] = "results/horizon_transfer_20261004/heldout_metrics.csv"
    run["outputs"]["summary"] = "results/horizon_transfer_20261004/summary.md"
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    print(dev_summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nHeld-out metrics and summary tables written.")


if __name__ == "__main__":
    main()
