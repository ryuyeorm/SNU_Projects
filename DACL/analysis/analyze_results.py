"""Statistical analysis and figures for gradient compatibility versus transfer.

The central inputs are one CSV row per source-target-seed measurement.  This script
computes Pearson and Spearman correlations, nonparametric bootstrap intervals, angular
sanity checks, and representative learning curves.  It reports undefined correlations
as JSON ``null`` when fewer than two nonconstant observations are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(x, y, statistic, seed=0, samples=10_000):
    """Estimate a percentile bootstrap confidence interval for a correlation.

    Args:
        x: One-dimensional compatibility observations.
        y: Corresponding transfer-benefit observations.
        statistic: Callable returning a scalar statistic from resampled ``x``/``y``.
        seed: Seed for reproducible bootstrap index samples.
        samples: Number of resampled datasets.

    Returns:
        Lower and upper 2.5-percentile bounds, or NaNs if no valid sample exists.
    """
    rng = np.random.default_rng(seed); values = []
    for _ in range(samples):
        ids = rng.integers(0, len(x), len(x))
        # Correlation is undefined when a bootstrap resample is constant.
        if np.std(x[ids]) > 0 and np.std(y[ids]) > 0:
            values.append(statistic(x[ids], y[ids]))
    return tuple(np.percentile(values, [2.5, 97.5])) if values else (np.nan, np.nan)


def _errorbar_by_angle(ax, df, column, ylabel):
    """Draw angle-group means with standard-error bars on a Matplotlib axis."""
    group = df.groupby("target_angle")[column]
    means, errors = group.mean(), group.sem().fillna(0)
    ax.errorbar(means.index, means, yerr=errors, marker="o", capsize=3)
    ax.set(xlabel="Target dynamics rotation (degrees)", ylabel=ylabel)


def _save(fig, output, name):
    """Finalize layout, save one PNG, and release its Matplotlib resources."""
    fig.tight_layout(); fig.savefig(output / name, dpi=160); plt.close(fig)


def analyze(results, output, bootstrap_samples=10_000):
    """Calculate the main statistics and generate all requested plots.

    Args:
        results: Path to the sweep CSV.
        output: Directory receiving figures and ``correlations.json``.
        bootstrap_samples: Resample count used for both correlation intervals.

    Returns:
        JSON-compatible dictionary of sample size, correlations, p-values, and CIs.
    """
    df = pd.read_csv(results); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    # The primary hypothesis uses the batchwise mean cosine and measured delta AUC.
    valid = df[["mean_cosine_alignment", "delta_auc"]].dropna()
    x, y = valid.iloc[:, 0].to_numpy(), valid.iloc[:, 1].to_numpy()
    # Both statistics require at least two observations with nonzero variance.
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        pearson = stats.pearsonr(x, y); spearman = stats.spearmanr(x, y)
        pci = bootstrap_ci(x, y, lambda a, b: stats.pearsonr(a, b).statistic, samples=bootstrap_samples)
        sci = bootstrap_ci(x, y, lambda a, b: stats.spearmanr(a, b).statistic, samples=bootstrap_samples)
    else:
        pearson = spearman = type("UndefinedCorrelation", (), {"statistic": np.nan, "pvalue": np.nan})()
        pci = sci = (np.nan, np.nan)
    # JSON has no portable NaN literal, so undefined values become null.
    clean = lambda value: float(value) if np.isfinite(value) else None
    summary = {"n": len(valid), "pearson_r": clean(pearson.statistic), "pearson_p_two_sided": clean(pearson.pvalue),
               "pearson_95pct_bootstrap_ci": [clean(v) for v in pci], "spearman_rho": clean(spearman.statistic),
               "spearman_p_two_sided": clean(spearman.pvalue),
               "spearman_95pct_bootstrap_ci": [clean(v) for v in sci]}
    summary["per_seed"] = {}
    for seed, seed_frame in df.groupby("seed"):
        sx = seed_frame["mean_cosine_alignment"].to_numpy()
        sy = seed_frame["delta_auc"].to_numpy()
        if len(seed_frame) >= 2 and np.std(sx) > 0 and np.std(sy) > 0:
            seed_p = stats.pearsonr(sx, sy)
            seed_s = stats.spearmanr(sx, sy)
            summary["per_seed"][str(int(seed))] = {
                "n": len(seed_frame), "pearson_r": clean(seed_p.statistic),
                "pearson_p_two_sided": clean(seed_p.pvalue),
                "spearman_rho": clean(seed_s.statistic),
                "spearman_p_two_sided": clean(seed_s.pvalue),
            }
    # Time-mean correlations use post-start information and are therefore descriptive,
    # not valid replacements for the pre-treatment primary predictor.
    for column in ("scratch_time_mean_alignment", "transfer_time_mean_alignment"):
        if column in df and len(df) >= 2 and df[column].std() > 0 and df["delta_auc"].std() > 0:
            dynamic_r = stats.pearsonr(df[column], df["delta_auc"])
            dynamic_s = stats.spearmanr(df[column], df["delta_auc"])
            prefix = column.removesuffix("_alignment")
            summary[f"descriptive_{prefix}_pearson_r"] = clean(dynamic_r.statistic)
            summary[f"descriptive_{prefix}_pearson_p_two_sided"] = clean(dynamic_r.pvalue)
            summary[f"descriptive_{prefix}_spearman_rho"] = clean(dynamic_s.statistic)
            summary[f"descriptive_{prefix}_spearman_p_two_sided"] = clean(dynamic_s.pvalue)
    (output / "correlations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Central result: early local compatibility against long-horizon transfer benefit.
    fig, ax = plt.subplots(); ax.scatter(x, y, alpha=.75)
    if len(x) >= 2 and np.std(x):
        slope, intercept = np.polyfit(x, y, 1); grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid)
    ax.axhline(0, color="grey", lw=.8); ax.axvline(0, color="grey", lw=.8)
    ax.set(xlabel="Gradient compatibility (mean cosine)", ylabel="Transfer benefit (delta AUC)")
    _save(fig, output, "alignment_vs_transfer.png")

    # Vertically stack each seed's result and the pooled result for direct comparison.
    seeds = sorted(df["seed"].unique())
    fig, axes = plt.subplots(len(seeds) + 1, 1, figsize=(7, 3.2 * (len(seeds) + 1)),
                             sharex=True, sharey=True)
    panels = [(f"Seed {int(seed)}", df[df["seed"] == seed]) for seed in seeds]
    panels.append(("Combined", df))
    for ax, (title, panel) in zip(axes, panels):
        px = panel["mean_cosine_alignment"].to_numpy()
        py = panel["delta_auc"].to_numpy()
        ax.scatter(px, py, alpha=.8)
        for _, row in panel.iterrows():
            ax.annotate(f"{row.target_angle:g}°", (row.mean_cosine_alignment, row.delta_auc),
                        xytext=(3, 3), textcoords="offset points", fontsize=7)
        if len(panel) >= 2 and np.std(px) > 0:
            slope, intercept = np.polyfit(px, py, 1)
            grid = np.linspace(px.min(), px.max(), 100)
            ax.plot(grid, intercept + slope * grid, lw=1)
        ax.axhline(0, color="grey", lw=.8); ax.axvline(0, color="grey", lw=.8)
        ax.set_title(title); ax.set_ylabel("Delta AUC")
    axes[-1].set_xlabel("Initial gradient compatibility")
    _save(fig, output, "alignment_vs_transfer_stacked_by_seed.png")

    for column, ylabel, name in [
        ("mean_cosine_alignment", "Gradient compatibility", "alignment_vs_target_angle.png"),
        ("delta_auc", "Transfer benefit (delta AUC)", "transfer_vs_target_angle.png")]:
        fig, ax = plt.subplots(); _errorbar_by_angle(ax, df, column, ylabel); _save(fig, output, name)
    for column, ylabel, name in [
        ("mean_cosine_alignment", "Gradient compatibility", "alignment_vs_angular_separation.png"),
        ("delta_auc", "Transfer benefit (delta AUC)", "transfer_vs_angular_separation.png")]:
        fig, ax = plt.subplots(); group = df.groupby("angular_separation")[column]
        means, errors = group.mean(), group.sem().fillna(0)
        ax.errorbar(means.index, means, yerr=errors, marker="o", capsize=3)
        ax.set(xlabel="Angular task separation (degrees)", ylabel=ylabel); _save(fig, output, name)

    # Select lowest, median, and highest compatibility rows for curve inspection.
    ordered = df.sort_values("mean_cosine_alignment")
    choices = [ordered.iloc[0], ordered.iloc[len(ordered) // 2], ordered.iloc[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, row, label in zip(axes, choices, ["negative/lowest", "neutral/median", "positive/highest"]):
        for key, name in [("scratch_curve", "scratch"), ("transfer_curve", "transfer")]:
            curve = json.loads(row[key]); ax.plot([p["step"] for p in curve], [p["return"] for p in curve], label=name)
        ax.set_title(f"{label}\n{row.source_angle:g}°→{row.target_angle:g}°, seed {int(row.seed)}")
        ax.set_xlabel("Training steps")
    axes[0].set_ylabel("Evaluation return"); axes[-1].legend(); _save(fig, output, "representative_curves.png")

    if "scratch_alignment_curve" in df and "transfer_alignment_curve" in df:
        # Show how compatibility evolves in both branches for every measured pair.
        ncols = min(4, len(df)); nrows = int(np.ceil(len(df) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows),
                                 sharex=True, sharey=True)
        axes = np.asarray(axes).reshape(-1)
        for ax, (_, row) in zip(axes, df.sort_values(["target_angle", "seed"]).iterrows()):
            for key, name in [("scratch_alignment_curve", "scratch"),
                              ("transfer_alignment_curve", "transfer")]:
                curve = json.loads(row[key])
                ax.plot([p["step"] for p in curve],
                        [p["mean_cosine_alignment"] for p in curve], label=name)
            ax.axhline(0, color="grey", lw=.8)
            ax.set_title(f"{row.source_angle:g}°→{row.target_angle:g}°, seed {int(row.seed)}")
        for ax in axes[len(df):]:
            ax.set_visible(False)
        axes[0].legend(); fig.supxlabel("Training steps")
        fig.supylabel("Gradient compatibility")
        _save(fig, output, "periodic_alignment_curves.png")
    return summary


def main():
    """Parse analysis CLI arguments, execute analysis, and print its summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--output", default="plots")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.results, args.output, args.bootstrap_samples), indent=2))


if __name__ == "__main__":
    main()
