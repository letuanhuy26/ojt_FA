from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/market_transfer_20261004_02"
MODEL_ORDER = [
    "regime_dynamic_gate_lstm",
    "dynamic_gate_lstm",
    "regime_as_feature_lstm",
    "random_forest",
]
MODEL_LABEL = {
    "regime_dynamic_gate_lstm": "RC-SFG",
    "dynamic_gate_lstm": "Dynamic gate, no regime",
    "regime_as_feature_lstm": "Regime as feature",
    "random_forest": "Random Forest",
}


def main() -> None:
    metrics = pd.read_csv(OUT / "fold_metrics.csv")
    bootstrap = pd.read_csv(OUT / "heldout_mcc_bootstrap.csv")
    run_path = OUT / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    expected_symbols = {"EWU", "EWJ"}
    if set(metrics["symbol"].unique()) != expected_symbols:
        raise SystemExit("Unexpected market set in transfer metrics.")
    if set(metrics["model"].unique()) != set(MODEL_ORDER):
        raise SystemExit("Unexpected model set in transfer metrics.")
    development = metrics[metrics["split"] == "development"]
    if (development.groupby(["symbol", "model"]).size() != 5).any():
        raise SystemExit("Each market/model must have exactly five development fold scores.")
    development_summary = development.groupby(
        ["region", "symbol", "model"], sort=False
    ).agg(
        folds=("fold_year", "count"),
        mean_mcc=("mcc", "mean"),
        sd_fold_mcc=("mcc", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_auroc=("auroc", "mean"),
        mean_positive_rate=("positive_rate", "mean"),
        mean_predicted_positive_rate=("predicted_positive_rate", "mean"),
        positive_folds=("mcc", lambda values: int((values > 0).sum())),
        n_forecasts=("n", "sum"),
    ).reset_index()
    development_summary.to_csv(OUT / "development_summary.csv", index=False, float_format="%.10g")

    heldout = metrics[metrics["split"].isin(["confirmation", "final"])][
        ["region", "symbol", "split", "model", "n", "positive_rate", "predicted_positive_rate",
         "mcc", "balanced_accuracy", "accuracy", "auroc", "brier", "log_loss"]
    ].copy()
    heldout.to_csv(OUT / "heldout_metrics.csv", index=False, float_format="%.10g")

    lines = [
        "# Cross-market configuration-transfer results (exploratory)",
        "",
        "The original SPY h=1 configuration was transferred unchanged to EWU and EWJ. Weights were refit on each asset's own past data; this tests hyperparameter/protocol transfer, not zero-shot transfer of learned model weights. No market-specific tuning was performed. All markets use the same Yahoo source and U.S.-listed ETF sessions; UK and Japan are country-equity proxies, not local-exchange index series.",
        "",
        "## Development mean MCC",
        "",
        "| Region | Symbol | Model | Mean MCC | SD across five folds | Positive folds | Mean positive-label rate | Mean predicted-positive rate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for region, symbol in [("United Kingdom", "EWU"), ("Japan", "EWJ")]:
        subset = development_summary[development_summary["symbol"] == symbol].set_index("model")
        for model in MODEL_ORDER:
            row = subset.loc[model]
            lines.append(
                f"| {region} | {symbol} | {MODEL_LABEL[model]} | {row['mean_mcc']:.5f} | "
                f"{row['sd_fold_mcc']:.5f} | {int(row['positive_folds'])}/5 | "
                f"{row['mean_positive_rate']:.3f} | {row['mean_predicted_positive_rate']:.3f} |"
            )
    lines.extend([
        "",
        "## Later-block MCC",
        "",
        "| Region | Symbol | Period | n | RC-SFG | Dynamic gate | Regime as feature | Random Forest |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for region, symbol in [("United Kingdom", "EWU"), ("Japan", "EWJ")]:
        for split in ["confirmation", "final"]:
            subset = heldout[(heldout["symbol"] == symbol) & (heldout["split"] == split)].set_index("model")
            row = subset.loc["regime_dynamic_gate_lstm"]
            mccs = [float(subset.loc[model, "mcc"]) for model in MODEL_ORDER]
            label = "Confirmation 2020–2021" if split == "confirmation" else "Final 2022–2025"
            lines.append(
                f"| {region} | {symbol} | {label} | {int(row['n'])} | "
                f"{mccs[0]:.5f} | {mccs[1]:.5f} | {mccs[2]:.5f} | {mccs[3]:.5f} |"
            )
    lines.extend([
        "",
        "## Paired RC-SFG minus Random Forest MCC",
        "",
        "| Region | Symbol | Period | Δ MCC | 95% circular block-bootstrap interval |",
        "|---|---|---|---:|---:|",
    ])
    for row in bootstrap.itertuples(index=False):
        label = "Confirmation 2020–2021" if row.split == "confirmation" else "Final 2022–2025"
        lines.append(
            f"| {row.region} | {row.symbol} | {label} | {row.point_delta_mcc:.5f} | "
            f"[{row.ci95_percentile_low:.5f}, {row.ci95_percentile_high:.5f}] |"
        )
    lines.extend([
        "",
        "Each interval uses 10,000 paired circular blocks of 20 target sessions, seed 1729; no p-values are computed. Results are post-hoc, use the same calendar blocks as the SPY pilot, and do not establish independent replication. Yahoo data rights remain unresolved. No economic simulation was run.",
        "",
        "Machine-readable outputs: `fold_metrics.csv`, `development_summary.csv`, `heldout_metrics.csv`, `heldout_mcc_bootstrap.csv`, and `run.json`. Raw prices, derived labels, and predictions are local-only.",
    ])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    run["outputs"]["development_summary"] = "results/market_transfer_20261004_02/development_summary.csv"
    run["outputs"]["heldout_metrics"] = "results/market_transfer_20261004_02/heldout_metrics.csv"
    run["outputs"]["summary"] = "results/market_transfer_20261004_02/summary.md"
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(development_summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nHeld-out results and summary tables written.")


if __name__ == "__main__":
    main()
