from __future__ import annotations

import itertools
from collections.abc import Iterable

from llmprefs.structs import TaskRecord, TaskType

Option = tuple[TaskRecord, ...]
Comparison = tuple[Option, Option]


def generate_comparisons(
    records: Iterable[TaskRecord],
    tasks_per_option: int,
) -> Iterable[Comparison]:
    """Generate comparisons of task options.

    Yield all permutations, e.g. both (A, B) and (B, A).
    The number of tasks in each option is determined by `tasks_per_option`.

    Some task types are treated differently:
    - opt out tasks always appear alone in an option, regardless of `tasks_per_option`.
    - opt out and free choice tasks are not compared to one another, only to other types
      of tasks
    """
    options = generate_options(records, tasks_per_option)
    comparisons = itertools.permutations(options, 2)
    yield from filter_comparisons(comparisons)


def generate_options(
    records: Iterable[TaskRecord],
    tasks_per_option: int,
) -> Iterable[Option]:
    """Generate all possible options of tasks.

    The number of tasks in each option is determined by `tasks_per_option`. However,
    that note that opt out tasks always appear alone in an option, regardless of
    `tasks_per_option`.
    """
    if tasks_per_option < 1:
        message = "tasks_per_option must be at least 1"
        raise ValueError(message)

    # Split records into regular and opt out tasks
    stream_a, stream_b = itertools.tee(records)
    regulars = itertools.filterfalse(is_opt_out_task, stream_a)
    opt_outs = filter(is_opt_out_task, stream_b)

    # Group regular tasks into options
    options = itertools.permutations(regulars, tasks_per_option)
    # Add opt out tasks by themselves
    yield from itertools.chain(
        options,
        ((oo,) for oo in opt_outs),
    )


def filter_comparisons(comparisons: Iterable[Comparison]) -> Iterable[Comparison]:
    """Remove comparisons of opt out or free choices tasks to themselves.

    Comparisons of opt out to free choice tasks are allowed.
    """
    for comp in comparisons:
        option_a, option_b = comp
        types_a = [rec.type for rec in option_a]
        types_b = [rec.type for rec in option_b]
        if types_a != types_b:
            yield comp
            continue

        regular_ids_a = [rec.id for rec in option_a if is_regular_task(rec)]
        regular_ids_b = [rec.id for rec in option_b if is_regular_task(rec)]
        if regular_ids_a != regular_ids_b:
            yield comp
            continue

        irregular_types_a = [rec.type for rec in option_a if not is_regular_task(rec)]
        irregular_types_b = [rec.type for rec in option_b if not is_regular_task(rec)]
        if irregular_types_a != irregular_types_b:
            yield comp

        # The regular tasks are identical, and the irregular types are the same.
        # Therefore, this comparison is redundant and we should drop it.


def is_opt_out_task(record: TaskRecord) -> bool:
    return record.type == TaskType.opt_out


def is_free_choice_task(record: TaskRecord) -> bool:
    return record.type == TaskType.free_choice


def is_regular_task(record: TaskRecord) -> bool:
    """Return True if the task is a regular task, i.e. not opt out or free choice."""
    return not is_opt_out_task(record) and not is_free_choice_task(record)
