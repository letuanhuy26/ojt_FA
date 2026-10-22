from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/development"
OUT = ROOT / "artifacts"


def main() -> None:
    signals = pd.read_csv(RESULTS / "gate_diagnostic_signal_series.csv", parse_dates=["signal_date"])
    fold = pd.read_csv(RESULTS / "gate_diagnostic_fold_summary.csv")
    seeds = pd.read_csv(RESULTS / "gate_diagnostic_seed_base_summary.csv")
    selected_seed = 1729
    signals = signals.loc[signals["model_seed_base"] == selected_seed]
    fold = fold.loc[fold["model_seed_base"] == selected_seed].set_index("fold_year")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(6, 1, figsize=(11.2, 12.8), constrained_layout=True)
    fig.suptitle("Exploratory development diagnostics (selected seed 1729)", fontsize=13, fontweight="bold")

    seed_names = [str(value) for value in seeds["model_seed_base"].tolist()]
    means = seeds["mean_mcc"].to_numpy()
    errors = seeds["sd_fold_mcc"].to_numpy()
    ax = axes[0]
    colors = ["#177e89" if value >= 0 else "#c44536" for value in means]
    ax.bar(seed_names, means, color=colors, alpha=0.85, width=0.62)
    ax.errorbar(seed_names, means, yerr=errors, fmt="none", ecolor="#283044", capsize=3, linewidth=1)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Mean fold MCC")
    ax.set_title("A. Development MCC by fixed model seed base (bars ± fold SD)", loc="left")
    ax.set_xlabel("Seed base")

    for axis, year in zip(axes[1:], range(2015, 2020)):
        data = signals.loc[signals["fold_year"] == year].sort_values("signal_date")
        stats = fold.loc[year]
        left = axis
        right = axis.twinx()
        left.plot(data["signal_date"], data["posterior_total_variation"], color="#7353ba", linewidth=0.8, alpha=0.9)
        right.plot(data["signal_date"], data["mean_absolute_gate_step"], color="#138a72", linewidth=0.8, alpha=0.8)
        left.axhline(float(stats["p90_posterior_tv"]), color="#7353ba", linestyle="--", linewidth=0.7, alpha=0.65)
        left.set_ylabel("Posterior TV", color="#62439f")
        right.set_ylabel("Gate step", color="#0e705c")
        left.set_title(
            f"{year}: mean max filtered-state probability={float(stats['mean_max_state_probability']):.3f}; "
            f"argmax switches={int(stats['argmax_state_switches'])}",
            loc="left",
        )
        left.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        left.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        if year == 2019:
            left.set_xlabel("Signal date (feature date; next session is the target)")
        left.grid(axis="x", alpha=0.15)

    fig.text(
        0.5,
        -0.006,
        "Post-hoc development-only analysis; seed 1729 was used in the original search. Error bars are fold SD, not confidence intervals.",
        ha="center",
        va="top",
        fontsize=8,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    png_path = OUT / "gate_diagnostic_development.png"
    svg_path = OUT / "gate_diagnostic_development.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")
    plt.close(fig)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
