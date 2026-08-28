#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import TASK_MATRICES
from experiments.run_pair import run_pair
from utils import load_config, metadata, save_json, unique_run_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--tasks", nargs="*", type=int, default=list(TASK_MATRICES))
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else config["experiment"]["seeds"]
    run_dir = unique_run_directory(config["paths"]["results"], "all_pairs")
    rows = []
    for seed in seeds:
        for source in args.tasks:
            for target in args.tasks:
                print(f"running source={source} target={target} seed={seed}", flush=True)
                row, curves = run_pair(source, target, seed, config)
                rows.append(row)
                save_json(run_dir / f"curves_source_{source}_target_{target}_seed_{seed}.json", curves)
                write_csv(run_dir / "all_pairs.csv", rows)
    save_json(run_dir / "metadata.json", metadata(config, seeds))
    print(run_dir / "all_pairs.csv")


def write_csv(path, rows):
    serial = [{key: json.dumps(value) if isinstance(value, (list, dict)) else value
               for key, value in row.items()} for row in rows]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=serial[0].keys())
        writer.writeheader(); writer.writerows(serial)


if __name__ == "__main__":
    main()
