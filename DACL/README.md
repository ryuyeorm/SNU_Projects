# Gradient compatibility as a predictor of policy transfer

This repository implements a controlled diagnostic study: does early actor-gradient
compatibility predict the measured AUC benefit of distilling a source actor–critic
policy into a target learner? It deliberately does **not** treat compatibility as task
similarity or use it as a transfer gate.

Every task has the identical fixed goal ``g = (1, 0)`` and reward ``-||s-g||``.
Task identity comes only from rotated action dynamics:

```text
s_next = s + delta * R(phi) * action + noise
```

Thus the task angle changes how a policy command moves the point, not which outcome
the environment rewards.

## Scientific design

For each source-target-seed record, a fresh target learner first gathers target-task
probe experience and then collects a fresh update-free on-policy diagnostic rollout.
On each chronological rollout batch the code computes, over the same target actor
parameters and the same states:

- the ordinary advantage actor–critic policy-gradient loss;
- the gradient of `KL(source || target)` toward a frozen source actor.

The reported compatibility is their cosine. Dot products, norms, batchwise mean and
standard deviation, cosine of averaged gradients, and low-norm confidence flags are
also retained. Diagnostic extraction uses `torch.autograd.grad` and performs no
optimizer step.

The task gradient uses normalized generalized advantage estimates (GAE) with a learned
state-value baseline. Advantages are detached, so only actor parameters enter the
diagnostic. The actor uses a tanh-squashed Gaussian. The distillation loss is the analytic KL
between their underlying, diagonal **pre-tanh Gaussians**. This is a deliberate first
version approximation: it is cheap, deterministic, and its gradient is exactly the
regularizer used in the transfer treatment. It is not the KL of the transformed
action distributions.

The prior loss is selectable with ``experiment.transfer_measure``:

- ``kl`` uses analytic ``KL(source || target)``;
- ``wasserstein`` uses smooth squared 2-Wasserstein distance
  ``||mu_s-mu_t||^2 + ||sigma_s-sigma_t||^2``.
- ``mean_mse`` uses only ``||mu_s-mu_t||^2`` and ignores policy standard deviations.

Both operate on pre-tanh diagonal Gaussians. The selected loss is used identically for
the diagnostic prior gradient and the branch treatment. Because their raw gradient
scales can differ, compare the recorded prior/task gradient-norm ratios and consider a
gradient-matched lambda ablation in addition to comparing them at the same lambda.

After probing, the full target agent (actor, value critic, optimizers, and RNG
snapshots) and diagnostic transition store are copied. Scratch and transfer branches
restore that same snapshot and receive equal interaction
and update counts. Their sole algorithmic difference is
`transfer_lambda * KL(source || target)` in the actor loss. Training updates use only
fresh chronological on-policy rollouts; the transition store is not replayed for
optimization. Deterministic evaluation
does not consume training RNG. AUC uses trapezoidal integration over evaluation return.

Because the score is local and first-order, only the empirical correlation with
`delta_auc = transfer_auc - scratch_auc` addresses the hypothesis.

## Setup and use

Python 3.10+ is required.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest
python experiments/run_sweep.py --config configs/default.yaml
python analysis/analyze_results.py --results results/rotated_dynamics_results.csv
```

The default sweep is 8 target dynamics rotations by 5 seeds. Edit the YAML for a quick smoke run
or a larger study. Source checkpoints are trained automatically when absent. A single
pair can be run with `experiments/measure_pair.py`.

Analysis writes Pearson and Spearman statistics with bootstrap confidence intervals,
the central compatibility-vs-transfer plot, dynamics-angle diagnostics,
and representative learning curves to `plots/`. The p-values are two-sided; the
pre-specified directional hypothesis is positive correlation and should be interpreted
with the confidence interval and scatter, not p-values alone.

## Reproducibility and scope

Python, NumPy, PyTorch, rollout sampling, environment, and action-space RNGs are seeded. Episode
starts use one convention per config (exact origin when `start_radius: 0`, otherwise a
uniform disk). Time-limit truncations permit critic bootstrapping. The config exposes
probe size, alignment batches, transfer strength, angles, seeds, learning rate,
transition noise, goal radius, and horizon. Initial compatibility is implemented;
periodic compatibility can be added at evaluation checkpoints without changing the
stored branch snapshot protocol.

## Comparing KL and Wasserstein

Run the matched five-seed configurations and then the paired comparison:

```bash
python3 experiments/run_sweep.py --config configs/five_seed_actor_critic.yaml
python3 experiments/run_sweep.py --config configs/five_seed_wasserstein.yaml
python3 experiments/run_sweep.py --config configs/five_seed_mean_mse.yaml
python3 analysis/compare_transfer_measures.py \
  --kl-results results/five_seed_rotated_dynamics.csv \
  --wasserstein-results results/five_seed_wasserstein.csv \
  --mean-mse-results results/five_seed_mean_mse.csv \
  --output plots/measure_comparison
```
