from collections.abc import Iterable
from pathlib import Path

from llmprefs.structs import ResultRecord


def save_results_jsonl(results: Iterable[ResultRecord], path: Path) -> None:
    """Save the results to a JSONL file.

    Append to the file if it already exists.
    """
    with path.open("a", encoding="utf-8") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")
