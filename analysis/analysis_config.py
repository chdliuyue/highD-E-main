"""Parameter configuration for the ``main_03_analysis`` entrypoint."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Sequence

TASK_CHOICES: FrozenSet[str] = frozenset(
    {
        "conflict_energy",
        "timeseries_coupling",
        "mec_baseline",
        "behavior_profiling",
    }
)

CONFLICT_TASK_CHOICES: FrozenSet[str] = frozenset(
    {
        "ghost_car",
        "timeseries",
        "phase_plane",
        "mec",
        "clusters",
        "map",
        "all",
    }
)


@dataclass(frozen=True)
class AnalysisCliDefaults:
    """Default values for command-line arguments."""

    tasks: str = "all"
    conflict_task: str = "all"
    recordings: str = "all"
    frame_rate: float = 25.0
    max_episodes: int = 1000
    output_root: Path = Path("output")


@dataclass
class AnalysisRunConfig:
    """Concrete run configuration after CLI parsing."""

    tasks: list[str]
    conflict_task: str
    recordings: list[int]
    mec_recordings: Sequence[int] | None
    frame_rate: float
    max_episodes: int
    output_root: Path


def parse_recording_list(rec_arg: str) -> list[int]:
    """Convert recording CLI input to an explicit list of ids."""
    if rec_arg.lower() == "all":
        return list(range(1, 61))
    return [int(x) for x in rec_arg.split(",") if x.strip()]


def parse_optional_recordings(rec_arg: str) -> list[int] | None:
    """Return ``None`` when all recordings are requested for MEC."""
    if rec_arg.lower() == "all":
        return None
    return [int(x) for x in rec_arg.split(",") if x.strip()]


def parse_tasks(task_arg: str) -> list[str]:
    """Validate requested task list against supported options."""
    requested = {t.strip().lower() for t in task_arg.split(",") if t.strip()}
    if not requested or "all" in requested:
        return sorted(TASK_CHOICES)
    unknown = requested.difference(TASK_CHOICES)
    if unknown:
        raise ValueError(f"Unknown task(s): {', '.join(sorted(unknown))}")
    return sorted(requested)


def parse_conflict_task(task: str) -> str:
    """Validate conflict-energy subtask selection."""
    task = task.lower()
    if task not in CONFLICT_TASK_CHOICES:
        raise ValueError(
            "Unknown conflict-energy subtask: "
            f"{task}. Choose from {', '.join(sorted(CONFLICT_TASK_CHOICES))}."
        )
    return task


def config_from_args(args: Namespace) -> AnalysisRunConfig:
    """Assemble a typed run configuration from parsed CLI arguments."""
    return AnalysisRunConfig(
        tasks=parse_tasks(args.tasks),
        conflict_task=parse_conflict_task(args.conflict_task),
        recordings=parse_recording_list(args.recordings),
        mec_recordings=parse_optional_recordings(args.recordings),
        frame_rate=args.frame_rate,
        max_episodes=args.max_episodes,
        output_root=args.output_root,
    )


__all__ = [
    "AnalysisCliDefaults",
    "AnalysisRunConfig",
    "CONFLICT_TASK_CHOICES",
    "TASK_CHOICES",
    "config_from_args",
    "parse_conflict_task",
    "parse_optional_recordings",
    "parse_recording_list",
    "parse_tasks",
]
