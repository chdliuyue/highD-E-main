"""Parallel execution helpers."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")


def run_parallel(func: Callable[[T], None], items: Iterable[T], num_workers: int) -> None:
    """Run a function over items using optional multiprocessing."""
    items_list: List[T] = list(items)
    if num_workers <= 1:
        for item in items_list:
            func(item)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for _ in executor.map(func, items_list):
                pass


__all__ = ["run_parallel"]
