# Heterogeneous Gradient Transfer

A controlled Actor–Critic experiment testing whether local source/target policy-gradient compatibility predicts the benefit of a fixed KL-transfer treatment across eight heterogeneous PointMass dynamics. Compatibility is diagnostic only: it never gates, selects, or changes transfer.

## Design commitments

- One fixed goal and reward for every task; only the registered 2×2 dynamics matrix changes.
- Tanh-squashed continuous Gaussian actor and value critic with separate optimizers.
- Pre-tanh Gaussian parameters define KL, squared W2, and mean-action MSE diagnostics.
- The predefined primary predictor is `C_mse_avg_grad`; the outcome is KL-transfer `delta_auc`.
- Identity pairs are retained as controls and excluded automatically from confirmatory analysis.
- Each target is normally adapted before diagnostics. Scratch and transfer branches clone actor, critic, optimizer, and RNG state; only fixed KL regularization differs.
- Source and target streams use `seed` and `seed + 10000`, respectively.

The repository did not contain a prior experiment, so `configs/default.yaml` records the fallback values from the study brief. In particular, the fallback `transfer_lambda: 0.1` is explicit and must be frozen before collecting confirmatory results.

## Setup

Run commands from this directory:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Tests and smoke validation

```bash
.venv/bin/pytest
python3 experiments/train_sources.py --config configs/smoke.yaml --tasks 0 --seeds 0
python3 experiments/run_pair.py --config configs/smoke.yaml --source 0 --target 0 --seed 0
```

The smoke configuration writes to `checkpoints_smoke/` and `results_smoke/`; it cannot
collide with full-study artifacts and cannot support scientific conclusions.

## Main experiment

```bash
python3 experiments/train_sources.py --config configs/default.yaml
python3 experiments/run_all_pairs.py --config configs/default.yaml --seeds 0 1 2 3 4
python3 analysis/analyze_results.py --results results/all_pairs_<timestamp>/all_pairs.csv
python3 analysis/plot_results.py --results results/all_pairs_<timestamp>/all_pairs.csv
```

Every all-pairs invocation creates a unique result directory and checkpoints are organized as `checkpoints/task_<id>/seed_<seed>.pt`. Full branch curves are separate JSON files. The CSV is checkpointed after every pair so an interrupted run retains completed records; rerunning intentionally creates a new experiment directory.
Source training refuses to overwrite a checkpoint unless `--overwrite` is supplied explicitly.

Only after validating the fixed-probe main experiment, run:

```bash
python3 experiments/run_probe_budget.py --config configs/default.yaml
python3 analysis/analyze_results.py --results results/probe_budget_<timestamp>/probe_budget.csv
python3 analysis/plot_results.py --results results/probe_budget_<timestamp>/probe_budget.csv
```

## Interpretation

The analysis reports pooled and within-seed correlations, source/target/seed fixed effects, raw-distance and zero-shot competitors, incremental R², and leave-one-source/target-out prediction. A null or negative finding is a valid outcome; do not tune tasks, choose a different compatibility statistic, or adjust per-pair transfer strength after inspecting results.
