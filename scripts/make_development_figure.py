from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "results/development/baseline_fold_metrics.csv"
SELECTED_PATH = ROOT / "autoresearch/model-20261004-01/final-best-fold-metrics.csv"
OUT_DIR = ROOT / "artifacts"


def main() -> None:
    baseline = pd.read_csv(BASELINE_PATH)
    selected = pd.read_csv(SELECTED_PATH)
    years = [2015, 2016, 2017, 2018, 2019]

    rf = baseline.loc[baseline["model"] == "random_forest"].set_index("fold_year")["mcc"]
    initial = baseline.loc[
        baseline["model"] == "regime_dynamic_gate_lstm"
    ].set_index("fold_year")["mcc"]
    tuned = selected.set_index("fold_year")["mcc"]

    series = [
        ("Random Forest", rf, "#355C7D"),
        ("Initial RC-SFG", initial, "#C77D24"),
        ("Selected RC-SFG", tuned, "#43866A"),
    ]
    x = np.arange(len(years), dtype=float)
    width = 0.24

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(8.8, 4.6), layout="constrained")
    for index, (label, values, color) in enumerate(series):
        ys = values.reindex(years).to_numpy(dtype=float)
        bars = ax.bar(
            x + (index - 1) * width,
            ys,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, rotation=90)

    ax.axhline(0, color="#343A40", linewidth=0.9, zorder=2)
    ax.set_xticks(x, [str(year) for year in years])
    ax.set_xlabel("Development target year")
    ax.set_ylabel("Matthews correlation coefficient (MCC)")
    ax.set_title("Development-fold MCC at the fixed 0.5 threshold")
    ax.set_ylim(-0.15, 0.18)
    ax.grid(axis="y", color="#DDE2E5", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.text(
        0.01,
        -0.015,
        "Five expanding-window folds; the selected RC-SFG was chosen on these same folds (single seed).",
        ha="left",
        va="top",
        fontsize=8,
        color="#495057",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "development_mcc_by_fold.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "development_mcc_by_fold.svg", bbox_inches="tight")
    plt.close(fig)
    print("Figure=artifacts/development_mcc_by_fold.png")
    print("Vector=artifacts/development_mcc_by_fold.svg")


if __name__ == "__main__":
    main()
