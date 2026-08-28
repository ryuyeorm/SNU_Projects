#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from agents import ActorCritic
from diagnostics import compute_alignment_batches, freeze_actor
from envs import get_task_matrix
from experiments.common import (auc, capture_rng_state, restore_rng_state, steps_to_success,
                                train_steps, train_with_evaluation)
from utils import evaluate, load_config, make_env, seed_everything


def load_source(source_task, seed, config):
    path = Path(config["paths"]["checkpoints"]) / f"task_{source_task}" / f"seed_{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing source checkpoint {path}; run train_sources.py first")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_agent_config = checkpoint.get("config")
    if checkpoint_agent_config is not None and checkpoint_agent_config != config["agent"]:
        raise ValueError(
            f"Source checkpoint {path} was trained with a different agent configuration. "
            f"checkpoint={checkpoint_agent_config}, requested={config['agent']}. "
            "Retrain the source experts with the same configuration used for the pair run; "
            "pass --overwrite to train_sources.py only if replacing this checkpoint is intended."
        )
    actor = ActorCritic(config["agent"]).actor
    try:
        actor.load_state_dict(checkpoint["actor"])
    except RuntimeError as exc:
        raise ValueError(
            f"Source checkpoint {path} is incompatible with the requested actor architecture. "
            "Retrain it using the same --config passed to this command."
        ) from exc
    return freeze_actor(actor), path


def run_pair(source_task, target_task, seed, config, probe_steps=None):
    source_actor, _ = load_source(source_task, seed, config)
    target_seed = int(seed) + 10_000
    seed_everything(target_seed, config["experiment"].get("deterministic_torch", True))
    zero_env = make_env(target_task, config["environment"], target_seed + 100)
    zero = evaluate(source_actor, zero_env, config["training"]["zero_shot_eval_episodes"],
                    target_seed + 200)
    learner = ActorCritic(config["agent"])
    probe_env = make_env(target_task, config["environment"], target_seed + 300)
    probe_steps = config["training"]["probe_steps"] if probe_steps is None else probe_steps
    observation = train_steps(learner, probe_env, probe_steps,
                              config["training"]["rollout_steps"])
    alignment_rollouts = []
    # Diagnostics collect target experience but never update the learner.
    for _ in range(config["training"]["num_alignment_batches"]):
        rollout, observation = learner.collect_rollout(
            probe_env, config["training"]["alignment_batch_steps"], observation)
        alignment_rollouts.append(rollout)
    diagnostic = compute_alignment_batches(learner.actor, source_actor, alignment_rollouts)
    branch_state = copy.deepcopy(learner.state_dict())
    rng_state = capture_rng_state()
    scratch = ActorCritic(config["agent"]); scratch.load_state_dict(copy.deepcopy(branch_state))
    transfer = ActorCritic(config["agent"]); transfer.load_state_dict(copy.deepcopy(branch_state))
    total = config["training"]["target_steps"]
    restore_rng_state(rng_state)
    scratch_curve = train_with_evaluation(scratch, target_task, config, target_seed + 400, total)
    restore_rng_state(rng_state)
    transfer_curve = train_with_evaluation(
        transfer, target_task, config, target_seed + 400, total, source_actor,
        config["training"]["transfer_lambda"])
    source_matrix, target_matrix = get_task_matrix(source_task), get_task_matrix(target_task)
    scratch_auc, transfer_auc = auc(scratch_curve), auc(transfer_curve)
    row = {
        "source_task": source_task, "target_task": target_task, "seed": seed,
        "source_matrix": source_matrix.tolist(), "target_matrix": target_matrix.tolist(),
        "is_identity_pair": source_task == target_task, "probe_steps": probe_steps,
        **diagnostic,
        "zero_shot_source_return": zero["mean_return"],
        "zero_shot_source_success_rate": zero["success_rate"],
        "matrix_frobenius_distance": float(np.linalg.norm(source_matrix - target_matrix, ord="fro")),
        "matrix_operator_distance": float(np.linalg.norm(source_matrix - target_matrix, ord=2)),
        "scratch_auc": scratch_auc, "transfer_auc": transfer_auc,
        "delta_auc": transfer_auc - scratch_auc,
        "scratch_early_return": scratch_curve[1]["mean_return"],
        "transfer_early_return": transfer_curve[1]["mean_return"],
        "early_return_difference": transfer_curve[1]["mean_return"] - scratch_curve[1]["mean_return"],
        "scratch_final_return": scratch_curve[-1]["mean_return"],
        "transfer_final_return": transfer_curve[-1]["mean_return"],
        "final_return_difference": transfer_curve[-1]["mean_return"] - scratch_curve[-1]["mean_return"],
        "scratch_steps_to_success": steps_to_success(scratch_curve),
        "transfer_steps_to_success": steps_to_success(transfer_curve),
        "scratch_max_return": max(x["mean_return"] for x in scratch_curve),
        "transfer_max_return": max(x["mean_return"] for x in transfer_curve),
    }
    row["max_return_difference"] = row["transfer_max_return"] - row["scratch_max_return"]
    curves = {"scratch": scratch_curve, "transfer": transfer_curve}
    return row, curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    row, _ = run_pair(args.source, args.target, args.seed, load_config(args.config))
    print(row)


if __name__ == "__main__":
    main()
