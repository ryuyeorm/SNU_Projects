"""Train and plot the gridworld risk-averse/risk-neutral comparison."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rqe-matplotlib")

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from rqe.algorithms.deep_rqe_ac import DeepRQEActorCritic
from rqe.training.trainer import (
    evaluate_gridworld,
    moving_average,
    train_gridworld,
)


PROBE_NAMES = (
    "start",
    "mutual_cooperation",
    "agent_0_defects",
    "agent_1_defects",
)
PROBE_STATES = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0],
        [4.0, 4.0, 4.0, 4.0],
        [0.0, 4.0, 4.0, 4.0],
        [4.0, 4.0, 4.0, 0.0],
    ]
)


@dataclass(frozen=True)
class RunTask:
    label: str
    risk_averse: bool
    run: int
    episodes: int
    batch_size: int
    hidden_dim: int
    replay_capacity: int
    log_interval: int
    evaluation_episodes: int
    device: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gridworld"),
    )
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
        args.evaluation_episodes = 2
        args.num_gpus = 1
    _validate_args(args)

    selected_device = _select_device(args.device)
    replay_capacity = (
        args.replay_capacity
        if args.replay_capacity is not None
        else args.episodes * 50
    )
    output_directory: Path = args.output_dir
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "runs").mkdir(exist_ok=True)

    print(f"using device: {selected_device}")
    print(f"replay capacity: {replay_capacity}")
    print(f"independent runs: {args.runs}")

    tasks = [
        RunTask(
            label=label,
            risk_averse=risk_averse,
            run=run,
            episodes=args.episodes,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            replay_capacity=replay_capacity,
            log_interval=args.log_interval,
            evaluation_episodes=args.evaluation_episodes,
            device=str(selected_device),
        )
        for risk_averse, label in (
            (True, "risk_averse"),
            (False, "risk_neutral"),
        )
        for run in range(args.runs)
    ]

    if selected_device.type == "cuda" and args.num_gpus > 1:
        completed = _run_multi_gpu(
            tasks,
            num_gpus=args.num_gpus,
            output_directory=output_directory,
        )
    else:
        completed = _run_sequential(tasks, output_directory)

    curves = {
        label: torch.stack(
            [completed[label][run]["social_welfare"] for run in range(args.runs)]
        )
        for label in ("risk_averse", "risk_neutral")
    }
    torch.save(curves, output_directory / "fig4_training_curves.pt")
    _save_diagnostics(completed, args.runs, output_directory)
    _print_verification_summary(completed, args.runs)
    _plot(curves, output_directory / "fig4.png")
    print(f"saved {output_directory / 'fig4.png'}")
    print(f"saved {output_directory / 'fig4_diagnostics.pt'}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0 or args.runs <= 0:
        raise ValueError("episodes and runs must be positive")
    if args.num_gpus <= 0:
        raise ValueError("num-gpus must be positive")
    if args.evaluation_episodes <= 0:
        raise ValueError("evaluation-episodes must be positive")
    if args.replay_capacity is not None and args.replay_capacity <= 0:
        raise ValueError("replay-capacity must be positive")


def _select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(
        "cuda" if requested == "cuda" or torch.cuda.is_available() else "cpu"
    )


def _run_sequential(
    tasks: list[RunTask],
    output_directory: Path,
) -> dict[str, dict[int, dict[str, Tensor]]]:
    completed = _empty_results()
    for task in tasks:
        result = _train_one(task)
        _record_result(completed, task, result, output_directory)
    return completed


def _run_multi_gpu(
    tasks: list[RunTask],
    num_gpus: int,
    output_directory: Path,
) -> dict[str, dict[int, dict[str, Tensor]]]:
    available = torch.cuda.device_count()
    if available < num_gpus:
        raise RuntimeError(
            f"requested {num_gpus} GPUs, but PyTorch sees only {available}"
        )

    context = mp.get_context("spawn")
    executors = [
        ProcessPoolExecutor(max_workers=1, mp_context=context)
        for _ in range(num_gpus)
    ]
    futures: dict[Future[dict[str, Tensor]], RunTask] = {}
    try:
        for task in tasks:
            gpu_id = task.run % num_gpus
            assigned = replace(task, device=f"cuda:{gpu_id}")
            future = executors[gpu_id].submit(_train_one, assigned)
            futures[future] = assigned

        completed = _empty_results()
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            _record_result(completed, task, result, output_directory)
        return completed
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def _empty_results() -> dict[str, dict[int, dict[str, Tensor]]]:
    return {"risk_averse": {}, "risk_neutral": {}}


def _record_result(
    completed: dict[str, dict[int, dict[str, Tensor]]],
    task: RunTask,
    result: dict[str, Tensor],
    output_directory: Path,
) -> None:
    completed[task.label][task.run] = result
    path = output_directory / "runs" / f"{task.label}_run_{task.run}.pt"
    torch.save(result, path)
    print(
        f"completed {task.label} run {task.run + 1}: saved {path}",
        flush=True,
    )


def _train_one(task: RunTask) -> dict[str, Tensor]:
    device = torch.device(task.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(task.run)
    print(
        f"[{task.label} run {task.run + 1} on {device}] starting",
        flush=True,
    )
    agent = DeepRQEActorCritic(
        observation_dim=4,
        action_dim=5,
        hidden_dim=task.hidden_dim,
        actor_learning_rate=5e-4,
        critic_learning_rate=5e-4,
        gamma=0.99,
        tau=5.0,
        epsilon=0.2,
        target_update=0.002,
        batch_size=task.batch_size,
        replay_capacity=task.replay_capacity,
        risk_averse=task.risk_averse,
        device=device,
    )
    training = train_gridworld(
        agent,
        episodes=task.episodes,
        seed=task.run,
        log_interval=task.log_interval,
        log_prefix=f"[{task.label} run {task.run + 1} {device}] ",
    )
    evaluation = evaluate_gridworld(
        agent,
        episodes=task.evaluation_episodes,
        seed=10_000 + task.run,
        deterministic=True,
    )
    actor_probabilities, adversary_probabilities = _probe_policies(agent)
    result = {
        "social_welfare": training.social_welfare,
        "agent_returns": training.agent_returns,
        "cooperation_rates": training.cooperation_rates,
        "defection_rates": training.defection_rates,
        "wandering_rates": (
            1.0 - training.cooperation_rates - training.defection_rates
        ),
        "evaluation_social_welfare": evaluation.social_welfare,
        "evaluation_agent_returns": evaluation.agent_returns,
        "evaluation_cooperation_rates": evaluation.cooperation_rates,
        "evaluation_defection_rates": evaluation.defection_rates,
        "evaluation_wandering_rates": (
            1.0 - evaluation.cooperation_rates - evaluation.defection_rates
        ),
        "probe_actor_probabilities": actor_probabilities,
        "probe_adversary_probabilities": adversary_probabilities,
    }
    del agent
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


@torch.no_grad()
def _probe_policies(agent: DeepRQEActorCritic) -> tuple[Tensor, Tensor]:
    states = PROBE_STATES.to(agent.compute_device)
    actors = torch.stack(
        [actor(states).softmax(dim=-1) for actor in agent.actors]
    ).cpu()
    if agent.risk_averse:
        adversaries = torch.stack(
            [adversary(states) for adversary in agent.adversaries]
        ).cpu()
    else:
        adversaries = torch.empty(0, len(PROBE_NAMES), agent.action_dim)
    return actors, adversaries


def _save_diagnostics(
    completed: dict[str, dict[int, dict[str, Tensor]]],
    runs: int,
    output_directory: Path,
) -> None:
    metric_names = tuple(next(iter(completed["risk_averse"].values())).keys())
    diagnostics = {
        "probe_names": PROBE_NAMES,
        **{
            metric: {
                label: torch.stack(
                    [completed[label][run][metric] for run in range(runs)]
                )
                for label in ("risk_averse", "risk_neutral")
            }
            for metric in metric_names
            if metric != "social_welfare"
        },
    }
    torch.save(diagnostics, output_directory / "fig4_diagnostics.pt")


def _print_verification_summary(
    completed: dict[str, dict[int, dict[str, Tensor]]],
    runs: int,
) -> None:
    for label in ("risk_averse", "risk_neutral"):
        final_social = torch.stack(
            [
                completed[label][run]["social_welfare"][-100:].mean()
                for run in range(runs)
            ]
        )
        evaluation_social = torch.stack(
            [
                completed[label][run]["evaluation_social_welfare"].mean()
                for run in range(runs)
            ]
        )
        cooperation = torch.stack(
            [
                completed[label][run]["cooperation_rates"][-100:].mean(dim=0)
                for run in range(runs)
            ]
        )
        defection = torch.stack(
            [
                completed[label][run]["defection_rates"][-100:].mean(dim=0)
                for run in range(runs)
            ]
        )
        classifications = [
            _classify_behavior(cooperation[run], defection[run])
            for run in range(runs)
        ]
        counts = {
            category: classifications.count(category)
            for category in sorted(set(classifications))
        }
        print(
            f"{label} verification: final MA100="
            f"{final_social.mean().item():.3f}±"
            f"{final_social.std(unbiased=False).item():.3f}, "
            f"deterministic evaluation="
            f"{evaluation_social.mean().item():.3f}±"
            f"{evaluation_social.std(unbiased=False).item():.3f}, "
            f"behaviors={counts}",
            flush=True,
        )


def _classify_behavior(cooperation: Tensor, defection: Tensor) -> str:
    if bool((cooperation > 0.5).all() and (defection < 0.1).all()):
        return "mutual_cooperation"
    if bool((defection > 0.5).all()):
        return "mutual_defection"
    if bool((defection > 0.5).any() and (cooperation > 0.5).any()):
        return "asymmetric"
    return "wandering_or_mixed"


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
