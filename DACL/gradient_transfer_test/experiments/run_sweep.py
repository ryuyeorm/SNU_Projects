"""Command-line driver for all configured dynamics rotations and seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gradient_transfer_test.experiments.common import load_config
from gradient_transfer_test.experiments.measure_pair import measure_pair
from gradient_transfer_test.experiments.train_source import train_source


def main():
    """Train/reuse the source, measure every pair, and incrementally save CSV.

    Incremental writes preserve completed records if a long research sweep is
    interrupted. Re-running currently starts a fresh table but reuses the source
    checkpoint when it already exists.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config); exp = config["experiment"]
    algorithm = config.get("algorithm", "sac")
    checkpoint_tag = ("actor_critic_gae_rotated_dynamics"
                      if algorithm == "actor_critic" else f"{algorithm}_rotated_dynamics")
    ckpt = Path(config["paths"]["checkpoints"]) / (
        f"source_{checkpoint_tag}_{exp['source_angle']:g}_seed_{exp['source_seed']}.pt"
    )
    # A single source prior is intentionally shared across all target conditions.
    if not ckpt.exists():
        print(f"Training source policy: {ckpt}", flush=True)
        train_source(config, output=ckpt)
    rows = []
    for angle in exp["target_angles"]:
        for seed in exp["seeds"]:
            print(f"Measuring {exp['source_angle']:g} -> {angle:g}, seed {seed}", flush=True)
            rows.append(measure_pair(config, ckpt, angle, seed))
            # Save after every expensive record instead of only at normal completion.
            output = Path(config["paths"]["results"])
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} records to {output}")


if __name__ == "__main__":
    main()
