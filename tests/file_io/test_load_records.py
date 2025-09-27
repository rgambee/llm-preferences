from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from llmprefs.file_io.load_records import load_records
from llmprefs.file_io.save_results import save_results_jsonl
from llmprefs.structs import ResultRecord, TaskRecord
from llmprefs.testing.factories import result_record_factory


class TestLoadTasks:
    def test_load_real_tasks(self) -> None:
        """Check that we can load the real tasks without errors."""
        path = Path(__file__).parent.parent.parent / "data" / "tasks.csv"
        tasks = load_records(path, TaskRecord)
        count = 0
        for record in tasks:
            count += 1
            assert record.task
        assert count > 0


class TestLoadResults:
    @pytest.mark.anyio
    async def test_load_results_jsonl(self) -> None:
        results = [result_record_factory(), result_record_factory()]
        with NamedTemporaryFile(suffix=".jsonl") as f:
            await save_results_jsonl(results, Path(f.name))
            loaded_results = tuple(load_records(Path(f.name), ResultRecord))
        assert len(loaded_results) == len(results)
        for loaded_result, result in zip(loaded_results, results, strict=True):
            assert loaded_result == result
