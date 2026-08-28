"""Train and save the frozen source policy used as a transfer prior."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gradient_transfer_test.agents.sac import ReplayBuffer
from gradient_transfer_test.experiments.common import interact_train, load_config, make_agent, make_env


def train_source(config, angle=None, seed=None, output=None):
    """Train one source-task actor and save its policy checkpoint.

    Args:
        config: Full parsed YAML configuration.
        angle: Optional source dynamics rotation overriding ``experiment.source_angle``.
        seed: Optional training seed overriding ``experiment.source_seed``.
        output: Optional exact checkpoint filename. When omitted, the configured
            checkpoint directory and a descriptive filename are used.

    Returns:
        :class:`Path` pointing to the saved checkpoint.
    """
    exp = config["experiment"]
    angle = exp["source_angle"] if angle is None else angle
    seed = exp["source_seed"] if seed is None else seed
    agent, env = make_agent(config, seed), make_env(config, angle)
    replay = ReplayBuffer(capacity=exp["replay_capacity"], seed=seed)
    interact_train(agent, env, replay, exp["source_train_steps"], exp["batch_size"],
                   exp["learning_starts"], seed)
    algorithm = config.get("algorithm", "sac")
    checkpoint_tag = ("actor_critic_gae_rotated_dynamics"
                      if algorithm == "actor_critic" else f"{algorithm}_rotated_dynamics")
    output = (Path(output) if output is not None else
              Path(config["paths"]["checkpoints"]) /
              f"source_{checkpoint_tag}_{angle:g}_seed_{seed}.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Only the actor is required as the frozen prior; architecture metadata rebuilds it.
    section = "actor_critic" if config.get("algorithm") == "actor_critic" else "sac"
    torch.save({"angle": angle, "seed": seed, "actor": agent.actor.state_dict(),
                "hidden_sizes": config[section]["hidden_sizes"],
                "algorithm": config.get("algorithm", "sac")}, output)
    return output


def main():
    """Parse command-line overrides and train one source policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--angle", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(train_source(load_config(args.config), args.angle, args.seed, args.output))


if __name__ == "__main__":
    main()
