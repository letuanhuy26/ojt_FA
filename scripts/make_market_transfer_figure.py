from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/market_transfer_20261004_02"
OUTPUT = ROOT / "artifacts/market_transfer_diagnostics"


def main() -> None:
    development = pd.read_csv(RESULTS / "development_summary.csv")
    heldout = pd.read_csv(RESULTS / "heldout_mcc_bootstrap.csv")
    model_order = [
        ("regime_dynamic_gate_lstm", "RC-SFG"),
        ("dynamic_gate_lstm", "Dynamic gate\n(no regime)"),
        ("regime_as_feature_lstm", "Regime as\nfeature"),
        ("random_forest", "Random Forest"),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.08, 1]})

    ax = axes[0]
    x = np.arange(len(model_order))
    width = 0.34
    for symbol, offset, color, label in [
        ("EWU", -width / 2, "#2563EB", "United Kingdom proxy · EWU"),
        ("EWJ", width / 2, "#7C3AED", "Japan proxy · EWJ"),
    ]:
        subset = development[development["symbol"] == symbol].set_index("model")
        means = [float(subset.loc[model, "mean_mcc"]) for model, _ in model_order]
        sds = [float(subset.loc[model, "sd_fold_mcc"]) for model, _ in model_order]
        ax.bar(x + offset, means, width=width, color=color, alpha=0.88, label=label)
        ax.errorbar(x + offset, means, yerr=sds, fmt="none", ecolor="#1F2937",
                    elinewidth=1.1, capsize=3, zorder=3)
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xticks(x, [label for _, label in model_order])
    ax.set_ylabel("Mean development MCC")
    ax.set_title("A. Frozen h=1 settings on development")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.28, "Error bars: SD across five annual folds, not confidence intervals.",
            transform=ax.transAxes, fontsize=8, color="#4B5563")

    ax = axes[1]
    rows = [
        ("EWU", "confirmation", "EWU · Confirmation 2020–2021", "#2563EB"),
        ("EWU", "final", "EWU · Final 2022–2025", "#2563EB"),
        ("EWJ", "confirmation", "EWJ · Confirmation 2020–2021", "#7C3AED"),
        ("EWJ", "final", "EWJ · Final 2022–2025", "#7C3AED"),
    ]
    ypos = np.arange(len(rows))[::-1]
    for y, (symbol, split, _, color) in zip(ypos, rows):
        row = heldout[(heldout["symbol"] == symbol) & (heldout["split"] == split)].iloc[0]
        estimate = float(row["point_delta_mcc"])
        low = float(row["ci95_percentile_low"])
        high = float(row["ci95_percentile_high"])
        ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]], fmt="o",
                    color=color, ecolor=color, capsize=4, linewidth=1.8, markersize=6)
    ax.axvline(0, color="#111827", linewidth=0.9)
    ax.set_yticks(ypos, [label for _, _, label, _ in rows])
    ax.set_xlim(-0.19, 0.22)
    ax.set_xlabel("Δ MCC (RC-SFG − Random Forest), 95% interval")
    ax.set_title("B. Paired circular-block bootstrap")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.28, "All four intervals include zero; 20-session blocks, 10,000 replicates.",
            transform=ax.transAxes, fontsize=8, color="#4B5563")

    fig.suptitle("Post-hoc transfer of the frozen SPY h=1 configuration", y=1.02, fontweight="bold")
    fig.text(0.5, -0.015,
             "EWU and EWJ are U.S.-listed country-equity ETF proxies; model weights are refit within each market.",
             ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved {OUTPUT.with_suffix('.png').relative_to(ROOT)}")
    print(f"Saved {OUTPUT.with_suffix('.svg').relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
