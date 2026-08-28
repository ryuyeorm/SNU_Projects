#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut


PRIMARY = "C_mse_avg_grad"
OUTCOME = "delta_auc"
PREDICTORS = {
    "Mean-MSE gradient alignment": "C_mse_avg_grad",
    "Wasserstein gradient alignment": "C_wasserstein_avg_grad",
    "KL gradient alignment": "C_kl_avg_grad",
    "Raw MSE distance": "raw_mse_distance",
    "Raw Wasserstein distance": "raw_wasserstein_distance",
    "Raw KL distance": "raw_kl_distance",
    "Zero-shot source return": "zero_shot_source_return",
    "Dynamics matrix distance": "matrix_frobenius_distance",
}


def correlations(frame, predictor):
    clean = frame[[predictor, OUTCOME]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2 or clean[predictor].nunique() < 2 or clean[OUTCOME].nunique() < 2:
        return np.nan, np.nan
    return pearsonr(clean[predictor], clean[OUTCOME]).statistic, spearmanr(clean[predictor], clean[OUTCOME]).statistic


def grouped_cv(frame, group_column, predictors):
    x = frame[predictors].to_numpy(); y = frame[OUTCOME].to_numpy()
    groups = frame[group_column].to_numpy(); predictions = np.full(len(frame), np.nan)
    splitter = LeaveOneGroupOut()
    for train, test in splitter.split(x, y, groups):
        predictions[test] = LinearRegression().fit(x[train], y[train]).predict(x[test])
    return {"rmse": float(mean_squared_error(y, predictions) ** 0.5),
            "prediction_correlation": float(pearsonr(y, predictions).statistic)}


def analyze(frame):
    frame = frame.loc[~frame["is_identity_pair"].astype(str).str.lower().isin(["true", "1"])].copy()
    pooled = correlations(frame, PRIMARY)
    within = []
    for seed, group in frame.groupby("seed"):
        pearson, spearman = correlations(group, PRIMARY)
        within.append({"seed": seed, "Pearson": pearson, "Spearman": spearman, "n_pairs": len(group)})
    fixed = smf.ols(f"{OUTCOME} ~ {PRIMARY} + C(source_task) + C(target_task) + C(seed)", data=frame).fit()
    comparison = [{"Predictor": label, "Pearson": correlations(frame, column)[0],
                   "Spearman": correlations(frame, column)[1]} for label, column in PREDICTORS.items()]
    zero = smf.ols(f"{OUTCOME} ~ zero_shot_source_return", data=frame).fit()
    zero_c = smf.ols(f"{OUTCOME} ~ zero_shot_source_return + {PRIMARY}", data=frame).fit()
    raw = smf.ols(f"{OUTCOME} ~ raw_mse_distance", data=frame).fit()
    raw_c = smf.ols(f"{OUTCOME} ~ raw_mse_distance + {PRIMARY}", data=frame).fit()
    return {"frame": frame, "pooled": pooled, "within": pd.DataFrame(within),
            "fixed": fixed, "comparison": pd.DataFrame(comparison),
            "incremental": pd.DataFrame([
                {"model": "zero_shot", "r_squared": zero.rsquared},
                {"model": "zero_shot+compatibility", "r_squared": zero_c.rsquared},
                {"model": "raw_mse", "r_squared": raw.rsquared},
                {"model": "raw_mse+compatibility", "r_squared": raw_c.rsquared}]),
            "loso": grouped_cv(frame, "source_task", [PRIMARY]),
            "loto": grouped_cv(frame, "target_task", [PRIMARY])}


def print_report(result):
    coefficient = result["fixed"].params[PRIMARY]; interval = result["fixed"].conf_int().loc[PRIMARY]
    print("Primary metric: mean-MSE avg-gradient cosine")
    print("Primary outcome: KL-transfer delta AUC")
    print(f"Number of non-identity records: {len(result['frame'])}")
    print(f"Pooled Pearson: {result['pooled'][0]:.6f}")
    print(f"Pooled Spearman: {result['pooled'][1]:.6f}")
    print(f"Mean within-seed Pearson: {result['within']['Pearson'].mean():.6f} "
          f"(SD {result['within']['Pearson'].std():.6f})")
    print(f"Fixed-effect beta_C: {coefficient:.6f}; SE {result['fixed'].bse[PRIMARY]:.6f}; "
          f"95% CI [{interval.iloc[0]:.6f}, {interval.iloc[1]:.6f}]; p={result['fixed'].pvalues[PRIMARY]:.6g}")
    print("\nWithin-seed analysis:\n", result["within"].to_string(index=False))
    print("\nDiagnostic comparison:\n", result["comparison"].to_string(index=False))
    print("\nIncremental value:\n", result["incremental"].to_string(index=False))
    print(f"\nLeave-one-source-out: {result['loso']}")
    print(f"Leave-one-target-out: {result['loto']}")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--results", required=True)
    args = parser.parse_args(); result = analyze(pd.read_csv(args.results)); print_report(result)
    output = Path(args.results).parent
    result["within"].to_csv(output / "within_seed.csv", index=False)
    result["comparison"].to_csv(output / "diagnostic_comparison.csv", index=False)
    result["incremental"].to_csv(output / "incremental_value.csv", index=False)
    with open(output / "fixed_effects.txt", "w", encoding="utf-8") as handle:
        handle.write(result["fixed"].summary().as_text())


if __name__ == "__main__":
    main()
