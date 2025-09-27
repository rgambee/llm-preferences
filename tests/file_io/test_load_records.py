from pathlib import Path

from llmprefs.file_io.load_records import load_records


class TestLoadTasks:
    def test_load_real_tasks(self) -> None:
        """Check that we can load the real tasks without errors."""
        path = Path(__file__).parent.parent.parent / "data" / "tasks.csv"
        tasks = load_records(path)
        count = 0
        for record in tasks:
            count += 1
            assert record.task
        assert count > 0
