"""Paired comparison of KL, squared-Wasserstein, and mean-MSE transfer."""

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


def correlation(frame):
    """Return Pearson and Spearman alignment/Delta-AUC summaries."""
    pearson = stats.pearsonr(frame.mean_cosine_alignment, frame.delta_auc)
    spearman = stats.spearmanr(frame.mean_cosine_alignment, frame.delta_auc)
    clean = lambda value: float(value) if np.isfinite(value) else None
    return {"n": len(frame), "pearson_r": clean(pearson.statistic),
            "pearson_p": clean(pearson.pvalue), "spearman_rho": clean(spearman.statistic),
            "spearman_p": clean(spearman.pvalue)}


def compare(kl_path, wasserstein_path, mean_mse_path, output):
    """Compare matched KL, W2, and mean-MSE records and generate plots.

    Args:
        kl_path: CSV generated with ``transfer_measure: kl``.
        wasserstein_path: CSV generated with ``transfer_measure: wasserstein``.
        mean_mse_path: CSV generated with ``transfer_measure: mean_mse``.
        output: Directory receiving JSON and figures.

    Returns:
        JSON-compatible combined statistical summary.
    """
    kl, w2, mse = (pd.read_csv(kl_path), pd.read_csv(wasserstein_path),
                    pd.read_csv(mean_mse_path))
    keys = ["source_angle", "target_angle", "seed"]
    paired = kl[keys + ["delta_auc"]].rename(columns={"delta_auc": "delta_auc_kl"})
    paired = paired.merge(
        w2[keys + ["delta_auc"]].rename(columns={"delta_auc": "delta_auc_wasserstein"}),
        on=keys, validate="one_to_one"
    ).merge(
        mse[keys + ["delta_auc"]].rename(columns={"delta_auc": "delta_auc_mean_mse"}),
        on=keys, validate="one_to_one"
    )

    def paired_summary(left, right):
        """Summarize matched right-minus-left transfer benefit."""
        difference = paired[f"delta_auc_{right}"] - paired[f"delta_auc_{left}"]
        test = stats.ttest_rel(paired[f"delta_auc_{right}"], paired[f"delta_auc_{left}"])
        clean = lambda value: float(value) if np.isfinite(value) else None
        return {"mean_difference": float(difference.mean()),
                "paired_t": clean(test.statistic), "paired_t_p": clean(test.pvalue),
                "right_better_count": int((difference > 0).sum())}

    summary = {
        "kl": correlation(kl), "wasserstein": correlation(w2),
        "mean_mse": correlation(mse),
        "matched_records": len(paired),
        "paired_transfer_comparisons": {
            "wasserstein_minus_kl": paired_summary("kl", "wasserstein"),
            "mean_mse_minus_kl": paired_summary("kl", "mean_mse"),
            "mean_mse_minus_wasserstein": paired_summary("wasserstein", "mean_mse"),
        },
    }
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "measure_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, frame, title in [(axes[0], kl, "KL"), (axes[1], w2, "Squared Wasserstein"),
                             (axes[2], mse, "Policy-mean MSE")]:
        ax.scatter(frame.mean_cosine_alignment, frame.delta_auc, alpha=.75)
        ax.axhline(0, color="grey", lw=.8); ax.axvline(0, color="grey", lw=.8)
        ax.set_title(title); ax.set_xlabel("Gradient compatibility")
    axes[0].set_ylabel("Delta AUC")
    fig.tight_layout(); fig.savefig(output / "measure_scatter_comparison.png", dpi=160)
    plt.close(fig)

    means = paired.groupby("target_angle")[[
        "delta_auc_kl", "delta_auc_wasserstein", "delta_auc_mean_mse"
    ]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(means.index, means.delta_auc_kl, marker="o", label="KL")
    ax.plot(means.index, means.delta_auc_wasserstein, marker="o", label="Squared W2")
    ax.plot(means.index, means.delta_auc_mean_mse, marker="o", label="Mean MSE")
    ax.axhline(0, color="grey", lw=.8); ax.legend()
    ax.set(xlabel="Target dynamics rotation (degrees)", ylabel="Mean Delta AUC")
    fig.tight_layout(); fig.savefig(output / "measure_transfer_by_angle.png", dpi=160)
    plt.close(fig)
    return summary


def main():
    """Parse paired result paths and print their comparison summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kl-results", required=True)
    parser.add_argument("--wasserstein-results", required=True)
    parser.add_argument("--mean-mse-results", required=True)
    parser.add_argument("--output", default="plots/measure_comparison")
    args = parser.parse_args()
    print(json.dumps(compare(args.kl_results, args.wasserstein_results,
                             args.mean_mse_results, args.output), indent=2))


if __name__ == "__main__":
    main()
