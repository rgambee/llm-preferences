from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_records(path: Path, record_type: type[T]) -> Iterable[T]:
    """Load the records from a CSV or JSON lines file.

    The column/field names and values must match those in the record type.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading records from {path}")
    if path.suffix == ".csv":
        return load_records_csv(path, record_type)
    if path.suffix == ".jsonl":
        return load_records_jsonl(path, record_type)
    logger.error(
        f"Unsupported file extension: {path.suffix}."
        + " Allowed extensions are .csv and .jsonl."
    )
    raise ValueError("Unsupported file extension")


def load_records_csv(path: Path, record_type: type[T]) -> Iterable[T]:
    """Load the records from a CSV file.

    The first row is assumed to be the header row. The column names must match those in
    the record type.
    """
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield record_type.model_validate(row)


def load_records_jsonl(path: Path, record_type: type[T]) -> Iterable[T]:
    """Load the records from a JSON lines file.

    The field names must match those in the record type.
    """
    with path.open("r") as f:
        for line in f:
            yield record_type.model_validate(json.loads(line))
