from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/horizon_transfer_20261004"
OUTPUT = ROOT / "artifacts/horizon_transfer_diagnostics"


def main() -> None:
    development = pd.read_csv(RESULTS / "development_summary.csv")
    heldout = pd.read_csv(RESULTS / "heldout_mcc_bootstrap.csv")
    models = [
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
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    x_positions = np.arange(len(models))
    bar_width = 0.34
    for horizon, offset, color in [(5, -bar_width / 2, "#2563EB"), (20, bar_width / 2, "#7C3AED")]:
        subset = development[development["horizon_sessions"] == horizon].set_index("model")
        means = [float(subset.loc[model, "mean_mcc"]) for model, _ in models]
        sds = [float(subset.loc[model, "sd_fold_mcc"]) for model, _ in models]
        ax.bar(x_positions + offset, means, width=bar_width, color=color, alpha=0.88,
               label=f"{horizon}-session target")
        ax.errorbar(x_positions + offset, means, yerr=sds, fmt="none", ecolor="#1F2937",
                    elinewidth=1.2, capsize=3, zorder=3)
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xticks(x_positions, [label for _, label in models])
    ax.set_ylabel("Mean development MCC")
    ax.set_title("A. Fixed h=1 settings on development")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.30, "Error bars: SD across five annual folds (not confidence intervals).",
            transform=ax.transAxes, fontsize=8, color="#4B5563")

    ax = axes[1]
    labels = [
        (5, "confirmation", "h=5 · Confirmation"),
        (5, "final", "h=5 · Final"),
        (20, "confirmation", "h=20 · Confirmation"),
        (20, "final", "h=20 · Final"),
    ]
    y = np.arange(len(labels))[::-1]
    for yi, (horizon, split, label) in zip(y, labels):
        row = heldout[(heldout["horizon_sessions"] == horizon) & (heldout["split"] == split)].iloc[0]
        estimate = float(row["point_delta_mcc"])
        low = float(row["ci95_percentile_low"])
        high = float(row["ci95_percentile_high"])
        color = "#2563EB" if horizon == 5 else "#7C3AED"
        ax.errorbar(estimate, yi, xerr=[[estimate - low], [high - estimate]], fmt="o",
                    color=color, ecolor=color, capsize=4, linewidth=1.8, markersize=6)
    ax.axvline(0, color="#111827", linewidth=0.9)
    ax.set_yticks(y, [label for _, _, label in labels])
    ax.set_xlim(-0.27, 0.27)
    ax.set_xlabel("Δ MCC (RC-SFG − Random Forest), 95% interval")
    ax.set_title("B. Paired circular-block bootstrap")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.30, "All intervals include zero; 20-origin blocks, 10,000 replicates.",
            transform=ax.transAxes, fontsize=8, color="#4B5563")

    fig.suptitle("Cross-horizon transfer of the frozen one-session configuration", y=1.02, fontweight="bold")
    fig.text(0.5, -0.02, "Post-hoc exploratory analysis on one provisional ETF; overlapping h-session targets; no horizon-specific tuning.",
             ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT.with_suffix(".png")
    svg_path = OUTPUT.with_suffix(".svg")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved {png_path.relative_to(ROOT)}")
    print(f"Saved {svg_path.relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
