# Disable line length limits
# ruff: noqa: E501
from enum import StrEnum
from textwrap import dedent
from typing import NewType

from pydantic import BaseModel

from llmprefs.task_structs import Comparison, Option

FormattedOption = NewType("FormattedOption", str)
FormattedComparison = NewType("FormattedComparison", str)


class TemplateStatus(StrEnum):
    enabled = "enabled"
    disabled = "disabled"


class ComparisonTemplate(BaseModel):
    id: int
    status: TemplateStatus
    template: str

    def format_comparison(self, comparison: Comparison) -> str:
        option_a, option_b = comparison
        formatted_option_a = format_option(option_a)
        formatted_option_b = format_option(option_b)
        return self.template.format(
            option_a=formatted_option_a,
            option_b=formatted_option_b,
        )


ALL_COMPARISON_TEMPLATES = (
    ComparisonTemplate(
        id=0,
        status=TemplateStatus.disabled,
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
        status=TemplateStatus.disabled,
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
        status=TemplateStatus.disabled,
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
    ComparisonTemplate(
        id=3,
        status=TemplateStatus.enabled,
        template=dedent(
            """
            Which option would you prefer to work on?

            Option A:
            {option_a}

            Option B:
            {option_b}

            After making your selection, you'll be given more details about how to get started.
            """
        ),
    ),
    ComparisonTemplate(
        id=4,
        status=TemplateStatus.enabled,
        template=dedent(
            """
            You have two options to choose from for your next assignment.

            Option A:
            {option_a}

            Option B:
            {option_b}

            What would you rather do?

            Note that you should not begin working on either option yet.
            """
        ),
    ),
    ComparisonTemplate(
        id=5,
        status=TemplateStatus.enabled,
        template=dedent(
            """
            You are a helpful assistant.

            As part of a beta program, you now have the ability to decide what to work on next.

            One possibility is working on option A:
            {option_a}

            Or you can work on option B:
            {option_b}

            Please make your decision now, after which you will get a more detailed description of the chosen option.
            """
        ),
    ),
    ComparisonTemplate(
        id=6,
        status=TemplateStatus.enabled,
        template=dedent(
            """
            Consider which of the following options you would prefer to work on:

            <option_a>
            {option_a}
            </option_a>

            <option_b>
            {option_b}
            </option_b>

            Reply with "A" or "B". After you make your decision, you'll be given more information about how to proceed with your chosen option.
            """
        ),
    ),
    ComparisonTemplate(
        id=7,
        status=TemplateStatus.enabled,
        template=dedent(
            """
            Of the options below, which would you prefer to complete?

            <option_a>
            {option_a}
            </option_a>

            <option_b>
            {option_b}
            </option_b>

            Answer with the letter of your selected option but do not begin working on it yet.
            """
        ),
    ),
)

ENABLED_COMPARISON_TEMPLATES = tuple(
    filter(
        lambda template: template.status == TemplateStatus.enabled,
        ALL_COMPARISON_TEMPLATES,
    )
)


def format_option(option: Option) -> FormattedOption:
    if len(option) == 1:
        return FormattedOption(option[0].task)
    steps = [f"{i + 1}. {task.task}" for i, task in enumerate(option)]
    return FormattedOption("\n".join(steps))
