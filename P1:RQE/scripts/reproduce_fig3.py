"""Reproduce the inspection-game policy dynamics shown in Figure 3."""

from pathlib import Path

import torch
from torch import Tensor

from rqe.algorithms.normal_form_gd import (
    solve_normal_form_game,
    solve_risk_neutral_game,
)
from rqe.envs.inspection_game import InspectionGame


def main() -> None:
    environment = InspectionGame(dtype=torch.float64)
    histories = {
        f"tau={tau}": solve_normal_form_game(
            environment.inspectee_payoffs,
            environment.inspector_payoffs,
            iterations=1_000,
            learning_rate=0.01,
            tau=tau,
            epsilon=0.2,
            regularizer="log_barrier",
        )
        for tau in (0.1, 0.5, 1.0, 2.0)
    }
    histories["risk-neutral"] = solve_risk_neutral_game(
        environment.inspectee_payoffs,
        environment.inspector_payoffs,
        iterations=1_000,
        learning_rate=0.01,
        epsilon=0.2,
        regularizer="log_barrier",
    )

    output_directory = Path("results/inspection")
    output_directory.mkdir(parents=True, exist_ok=True)
    data = {
        f"{label}_{player}": history[player][:, 0]
        for label, history in histories.items()
        for player in ("policy_1", "policy_2")
    }
    torch.save(data, output_directory / "fig3_policy_dynamics.pt")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _print_final_policies(histories)
        print("matplotlib is unavailable; saved tensor data only")
        return

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for label, history in histories.items():
        axes[0].plot(history["policy_1"][:, 0].numpy(), label=label)
        axes[1].plot(history["policy_2"][:, 0].numpy(), label=label)
    axes[0].set_title("Player 1 Policy")
    axes[1].set_title("Player 2 Policy")
    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Probability of action 0")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_directory / "fig3.png", dpi=200)
    plt.close(figure)
    _print_final_policies(histories)


def _print_final_policies(histories: dict[str, dict[str, Tensor]]) -> None:
    for label, history in histories.items():
        player_1 = history["policy_1"][-1].tolist()
        player_2 = history["policy_2"][-1].tolist()
        print(f"{label:>12}: player 1={player_1}, player 2={player_2}")


if __name__ == "__main__":
    main()
