#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agents import ActorCritic
from envs import TASK_MATRICES
from experiments.common import train_with_evaluation
from utils import load_config, metadata, save_json, seed_everything


def train_source(task_id, seed, config, overwrite=False):
    source_seed = int(seed)
    seed_everything(source_seed, config["experiment"].get("deterministic_torch", True))
    agent = ActorCritic(config["agent"])
    curve = train_with_evaluation(agent, task_id, config, source_seed,
                                  config["training"]["source_steps"])
    root = Path(config["paths"]["checkpoints"])
    checkpoint = root / f"task_{task_id}" / f"seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {checkpoint}; pass --overwrite explicitly")
    torch.save({**agent.state_dict(), "task_id": task_id, "seed": seed,
                "final_evaluation": curve[-1], "training_curve": curve,
                "metadata": metadata(config, seed)}, checkpoint)
    save_json(checkpoint.with_suffix(".curve.json"), curve)
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--tasks", nargs="*", type=int, default=list(TASK_MATRICES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else config["experiment"]["seeds"]
    for seed in seeds:
        for task_id in args.tasks:
            print(train_source(task_id, seed, config, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
