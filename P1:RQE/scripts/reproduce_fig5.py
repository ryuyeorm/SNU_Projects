"""Plot the paper's MAPPO and MADDPG gridworld baseline comparison."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rqe-matplotlib")

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from rqe.training.trainer import moving_average


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Figure 5 from separately generated MAPPO and MADDPG "
            "social-welfare tensors. Each input must have shape [runs, episodes]."
        )
    )
    parser.add_argument("--mappo-data", type=Path, required=True)
    parser.add_argument("--maddpg-data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gridworld/fig5.png"),
    )
    args = parser.parse_args()

    curves = {
        "MAPPO": _load_curves(args.mappo_data),
        "MADDPG": _load_curves(args.maddpg_data),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot(curves, args.output)
    print(f"saved {args.output}")


def _load_curves(path: Path) -> Tensor:
    if not path.exists():
        raise FileNotFoundError(f"baseline result file does not exist: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        if "social_welfare" not in data:
            raise ValueError(f"{path} must contain a 'social_welfare' tensor")
        data = data["social_welfare"]
    if not isinstance(data, Tensor) or data.ndim != 2:
        raise ValueError(f"{path} must contain a [runs, episodes] tensor")
    if not torch.isfinite(data).all():
        raise ValueError(f"{path} contains non-finite values")
    return data.float()


def _plot(curves: dict[str, Tensor], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, (name, values) in zip(axes, curves.items()):
        smoothed = torch.stack([moving_average(run, 100) for run in values])
        for run in smoothed:
            axis.plot(run.numpy(), alpha=0.55, linewidth=1)
        axis.plot(
            smoothed.mean(dim=0).numpy(),
            color="black",
            linewidth=2,
            label="run mean",
        )
        axis.set_title(name)
        axis.set_xlabel("Episode")
        axis.legend()
    axes[0].set_ylabel("MA100 Social Welfare")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
