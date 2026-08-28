#!/usr/bin/env python
"""Run the explicitly secondary probe-budget experiment."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import TASK_MATRICES
from experiments.run_all_pairs import write_csv
from experiments.run_pair import run_pair
from utils import load_config, save_json, unique_run_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--tasks", nargs="*", type=int, default=list(TASK_MATRICES))
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else config["experiment"]["seeds"]
    output = unique_run_directory(config["paths"]["results"], "probe_budget")
    rows = []
    for budget in config["training"]["probe_budgets"]:
        for seed in seeds:
            for source in args.tasks:
                for target in args.tasks:
                    row, curves = run_pair(source, target, seed, config, probe_steps=budget)
                    rows.append(row); write_csv(output / "probe_budget.csv", rows)
                    save_json(output / f"curves_n{budget}_s{source}_t{target}_seed{seed}.json", curves)
    print(output / "probe_budget.csv")


if __name__ == "__main__":
    main()
