from typing import NewType

from pydantic import BaseModel

from llmprefs.comparisons import Comparison, Option

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


def format_option(option: Option) -> FormattedOption:
    if len(option) == 1:
        return FormattedOption(option[0].task)
    steps = [f"{i + 1}. {task.task}" for i, task in enumerate(option)]
    return FormattedOption("\n".join(steps))
