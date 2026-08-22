"""Train and plot the gridworld risk-averse/risk-neutral comparison."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rqe-matplotlib")

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from rqe.algorithms.deep_rqe_ac import DeepRQEActorCritic
from rqe.training.trainer import moving_average, train_gridworld


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1_000)
    args = parser.parse_args()
    if args.quick:
        args.episodes = 10
        args.runs = 1
        args.batch_size = 64
        args.hidden_dim = 32
        args.log_interval = 5

    curves: dict[str, list[Tensor]] = {
        "risk_averse": [],
        "risk_neutral": [],
    }
    output_directory = Path("results/gridworld")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = None if args.device == "auto" else args.device
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"using device: {selected_device}")

    for risk_averse, label in ((True, "risk_averse"), (False, "risk_neutral")):
        for run in range(args.runs):
            seed = run
            print(f"training {label} run {run + 1}/{args.runs}")
            torch.manual_seed(seed)
            agent = DeepRQEActorCritic(
                observation_dim=4,
                action_dim=5,
                hidden_dim=args.hidden_dim,
                actor_learning_rate=5e-4,
                critic_learning_rate=5e-4,
                gamma=0.99,
                tau=5.0,
                epsilon=0.2,
                target_update=0.002,
                batch_size=args.batch_size,
                replay_capacity=100_000,
                risk_averse=risk_averse,
                device=selected_device,
            )
            result = train_gridworld(
                agent,
                episodes=args.episodes,
                seed=seed,
                log_interval=args.log_interval,
            )
            curves[label].append(result.social_welfare)
            torch.save(
                {name: torch.stack(values) for name, values in curves.items() if values},
                output_directory / "fig4_training_curves.pt",
            )

    stacked = {name: torch.stack(values) for name, values in curves.items()}
    _plot(stacked, output_directory / "fig4.png")
    print(f"saved {output_directory / 'fig4.png'}")


def _plot(curves: dict[str, Tensor], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = (("risk_averse", "Risk-Averse"), ("risk_neutral", "Risk-Neutral"))
    for axis, (key, title) in zip(axes, panels):
        smoothed = torch.stack(
            [moving_average(run, 100) for run in curves[key]]
        )
        for run in smoothed:
            axis.plot(run.numpy(), alpha=0.45, linewidth=1)
        axis.plot(
            smoothed.mean(dim=0).numpy(),
            color="black",
            linewidth=2,
            label="run mean",
        )
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.legend()
    axes[0].set_ylabel("MA100 Social Welfare")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
