from pathlib import Path

from llmprefs.load_dataset import load_dataset


class TestLoadDataset:
    def test_load_real_dataset(self) -> None:
        """Check that we can load the real dataset without errors."""
        path = Path(__file__).parent.parent / "data" / "tasks.csv"
        dataset = load_dataset(path)
        count = 0
        for record in dataset:
            count += 1
            assert record.task
        assert count > 0
