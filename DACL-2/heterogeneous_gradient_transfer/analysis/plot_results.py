#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def scatter(frame, x, title, path):
    figure, axis = plt.subplots(figsize=(6, 4.5))
    sns.regplot(data=frame, x=x, y="delta_auc", scatter_kws={"alpha": 0.55}, ax=axis)
    axis.set_title(title); figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def heatmap(frame, value, title, path):
    matrix = frame.groupby(["source_task", "target_task"])[value].mean().unstack()
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, cmap="coolwarm", center=0, annot=True, fmt=".2g", ax=axis)
    axis.set_title(title); figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--results", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(); frame = pd.read_csv(args.results)
    output = Path(args.output or Path(args.results).parent / "plots"); output.mkdir(parents=True, exist_ok=True)
    non_identity = frame.loc[~frame["is_identity_pair"].astype(str).str.lower().isin(["true", "1"])]
    scatter(non_identity, "C_mse_avg_grad", "Primary: compatibility vs transfer", output / "main_result.png")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (column, name) in zip(axes, [("C_mse_avg_grad", "Mean MSE"),
                                           ("C_wasserstein_avg_grad", "Wasserstein"),
                                           ("C_kl_avg_grad", "KL")]):
        sns.regplot(data=non_identity, x=column, y="delta_auc", ax=axis,
                    scatter_kws={"alpha": 0.45}); axis.set_title(name)
    figure.tight_layout(); figure.savefig(output / "diagnostic_comparison.png", dpi=180); plt.close(figure)
    heatmap(frame, "delta_auc", "Mean transfer delta AUC", output / "transfer_matrix.png")
    heatmap(frame, "C_mse_avg_grad", "Mean compatibility", output / "compatibility_matrix.png")
    scatter(non_identity, "zero_shot_source_return", "Zero-shot baseline", output / "zero_shot.png")
    scatter(non_identity, "raw_mse_distance", "Raw MSE-distance baseline", output / "raw_mse.png")
    if "probe_steps" in frame and frame["probe_steps"].nunique() > 1:
        correlations = frame.groupby("probe_steps").apply(
            lambda group: group["C_mse_avg_grad"].corr(group["delta_auc"]), include_groups=False)
        figure, axis = plt.subplots(figsize=(6, 4)); correlations.plot(marker="o", ax=axis)
        axis.set(xlabel="Probe environment steps", ylabel="Pearson r", title="Probe sample efficiency")
        figure.tight_layout(); figure.savefig(output / "probe_sample_efficiency.png", dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
