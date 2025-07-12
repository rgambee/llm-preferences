import json
from collections.abc import Iterable
from pathlib import Path

from structs import TaskRecord


def load_dataset(path: Path) -> Iterable[TaskRecord]:
    """Load the dataset from a JSON lines file.

    The columns names and values must match those in TaskRecord.
    """
    with path.open("r") as f:
        for line in f:
            yield TaskRecord.model_validate(json.loads(line))
