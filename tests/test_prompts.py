from llmprefs.prompts import format_option
from llmprefs.structs import TaskType
from llmprefs.testing.factories import task_record_factory


class TestFormatOption:
    def test_empty_option(self) -> None:
        formatted_option = format_option(())
        assert formatted_option == ""

    def test_single_task(self) -> None:
        option = task_record_factory([TaskType.dummy])
        formatted_option = format_option(option)
        assert formatted_option == option[0].task

    def test_multiple_tasks(self) -> None:
        option = task_record_factory([TaskType.dummy, TaskType.dummy])
        formatted_option = format_option(option)
        assert formatted_option == f"1. {option[0].task}\n2. {option[1].task}"
