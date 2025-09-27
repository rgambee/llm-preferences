import csv
import json
import logging
from collections.abc import Iterable
from pathlib import Path

from llmprefs.structs import TaskRecord


def load_dataset(path: Path) -> Iterable[TaskRecord]:
    """Load the dataset from a CSV or JSON lines file.

    The column/field names and values must match those in TaskRecord.
    """
    if path.suffix == ".csv":
        return load_dataset_csv(path)
    if path.suffix == ".jsonl":
        return load_dataset_jsonl(path)
    logging.getLogger(__name__).error(
        f"Unsupported file extension: {path.suffix}. "
        "Allowed extensions are .csv and .jsonl."
    )
    raise ValueError("Unsupported file extension")


def load_dataset_csv(path: Path) -> Iterable[TaskRecord]:
    """Load the dataset from a JSON lines file.

    The first row is assumed to be the header row. The column names and values must
    match those in TaskRecord.
    """
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield TaskRecord.model_validate(row)


def load_dataset_jsonl(path: Path) -> Iterable[TaskRecord]:
    """Load the dataset from a JSON lines file.

    The field names and values must match those in TaskRecord.
    """
    with path.open("r") as f:
        for line in f:
            yield TaskRecord.model_validate(json.loads(line))
