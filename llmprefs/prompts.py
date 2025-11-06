# Disable line length limits
# ruff: noqa: E501
from textwrap import dedent
from typing import NewType

from pydantic import BaseModel

from llmprefs.task_structs import Comparison, Option

FormattedOption = NewType("FormattedOption", str)
FormattedComparison = NewType("FormattedComparison", str)


class ComparisonTemplate(BaseModel):
    id: int
    template: str

    def format_comparison(self, comparison: Comparison) -> str:
        option_a, option_b = comparison
        formatted_option_a = format_option(option_a)
        formatted_option_b = format_option(option_b)
        return self.template.format(
            option_a=formatted_option_a,
            option_b=formatted_option_b,
        )


COMPARISON_TEMPLATES = (
    ComparisonTemplate(
        id=0,
        template=dedent(
            """
            Which option would you prefer to work on?

            Option A:
            {option_a}

            Option B:
            {option_b}
            """
        ),
    ),
    ComparisonTemplate(
        id=1,
        template=dedent(
            """
            You have two options to choose from for your next assignment.

            Option A:
            {option_a}

            Option B:
            {option_b}

            What would you rather do?
            """
        ),
    ),
    ComparisonTemplate(
        id=2,
        template=dedent(
            """
            You are a helpful assistant.

            As part of a beta program, you now have the ability to decide what to work on next.

            One possibility is working on option A:
            {option_a}

            Or you can work on option B:
            {option_b}

            Please make your decision now.
            """
        ),
    ),
)


def format_option(option: Option) -> FormattedOption:
    if len(option) == 1:
        return FormattedOption(option[0].task)
    steps = [f"{i + 1}. {task.task}" for i, task in enumerate(option)]
    return FormattedOption("\n".join(steps))
