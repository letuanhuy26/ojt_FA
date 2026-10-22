from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/development/economic_sensitivity_summary.csv"
OUT_PNG = ROOT / "artifacts/economic_sensitivity_development.png"
OUT_SVG = ROOT / "artifacts/economic_sensitivity_development.svg"


def main() -> None:
    frame = pd.read_csv(RESULTS)
    if set(frame["model"].unique()) != {
        "selected_rc_sfg_seed_1729",
        "random_forest",
        "always_long",
    }:
        raise SystemExit("Unexpected economic sensitivity model set.")
    if set(frame["per_side_cost_bps"].unique()) != {0.0, 1.0, 2.5, 5.0, 10.0}:
        raise SystemExit("Unexpected per-side cost scenarios.")

    style = {
        "selected_rc_sfg_seed_1729": ("RC-SFG (selected, seed 1729)", "#2457A7"),
        "random_forest": ("Random Forest", "#C65D26"),
        "always_long": ("Always long", "#4C7A63"),
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.20, top=0.80)
    for model, (label, color) in style.items():
        series = frame.loc[frame["model"] == model].sort_values("per_side_cost_bps")
        ax.plot(
            series["per_side_cost_bps"],
            100.0 * series["net_total_return"],
            marker="o",
            markersize=5,
            linewidth=2.0,
            color=color,
            label=label,
        )

    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.text(
        0.0,
        1.20,
        "Development-only net return sensitivity",
        transform=ax.transAxes,
        fontsize=15,
        weight="bold",
        color="#20252B",
        va="bottom",
    )
    ax.text(
        0.0,
        1.11,
        "SPY open-to-close long-or-cash · 2015–2019 · post-hoc, n = 1,258",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="bottom",
    )
    ax.set_xlabel("Assumed transaction cost (basis points per side)")
    ax.set_ylabel("Compounded total return (%)")
    ax.set_xticks([0, 1, 2.5, 5, 10], labels=["0", "1", "2.5", "5", "10"])
    ax.set_xlim(-0.3, 10.5)
    ax.set_ylim(-100, 42)
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.text(
        0.13,
        0.035,
        "Entry at next open; exit same-session close. Fixed cost applied on entry and exit; excludes spread, impact, taxes and fill risk.",
        fontsize=8,
        color="#555555",
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}")
    print(f"Wrote {OUT_SVG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
