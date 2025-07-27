import pytest

from llmprefs.prompts import COMPARISON_TEMPLATES, ComparisonTemplate, format_option
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


class TestComparisonTemplate:
    def test_valid_template(self) -> None:
        option_a, option_b = task_record_factory([TaskType.dummy] * 2)
        comparison = ((option_a,), (option_b,))
        template = ComparisonTemplate(
            id=0,
            template="Which would you prefer?\n{option_a}\n\n{option_b}",
        )
        formatted = template.format_comparison(comparison)
        assert option_a.task in formatted
        assert option_b.task in formatted

    def test_invalid_template(self) -> None:
        option_a, option_b = task_record_factory([TaskType.dummy] * 2)
        comparison = ((option_a,), (option_b,))
        template = ComparisonTemplate(
            id=0,
            template="Which would you prefer?\n{0}\n\n{1}",
        )
        with pytest.raises(
            IndexError,
            match="Replacement index 0 out of range for positional args tuple",
        ):
            template.format_comparison(comparison)


class TestTemplateInstances:
    @pytest.mark.parametrize("template", COMPARISON_TEMPLATES)
    def test_template_valid(self, template: ComparisonTemplate) -> None:
        option_a, option_b = task_record_factory([TaskType.dummy] * 2)
        comparison = ((option_a,), (option_b,))
        template.format_comparison(comparison)

    @pytest.mark.parametrize("template", COMPARISON_TEMPLATES)
    def test_indentation(self, template: ComparisonTemplate) -> None:
        option_a, option_b = task_record_factory([TaskType.dummy] * 2)
        comparison = ((option_a,), (option_b,))
        formatted = template.format_comparison(comparison)
        for line in formatted.split("\n"):
            assert line == line.strip()
