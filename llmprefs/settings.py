from pathlib import Path

from openai.types.shared_params.reasoning_effort import ReasoningEffort
from pydantic import Field
from pydantic_settings import BaseSettings

from llmprefs.api.structs import LLM


class Settings(
    BaseSettings,
    cli_parse_args=True,
    cli_enforce_required=True,
    cli_kebab_case=True,
    cli_implicit_flags=True,
):
    # IO Settings
    input_path: Path = Field(
        default=...,
        description="Path to input file containing task dataset",
    )
    output_path: Path = Field(
        default=...,
        description="""
            Path to output file for saving results.
            If it already exists, new results will be appended.
        """,
    )
    log_file: Path | None = Field(
        default=None,
        description="Path to log file. By default, only log to the console.",
    )

    # Task Settings
    tasks_per_option: int = Field(
        default=2,
        description="Number of tasks per option",
    )
    samples_per_comparison: int = Field(
        default=1,
        description="Number of answers to generate for each comparison",
    )
    count_comparisons_only: bool = Field(
        default=False,
        description="Only count the number of comparisons, don't generate any results",
    )

    # API settings
    concurrent_requests: int = Field(
        default=10,
        description="Maximum number of concurrent API requests",
    )
    model: LLM = Field(
        default=...,
        description="Model to use",
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature for the model",
    )
    max_output_tokens: int = Field(
        default=1000,
        description="Maximum number of output tokens per response",
    )
    system_prompt: str = Field(
        default="",
        description="System prompt for the model",
    )
    structured_output: bool = Field(
        default=False,
        description="Whether to enforce structured output",
    )

    # Anthropic settings
    anthropic_thinking_budget: int = Field(
        default=0,
        description="""
            Maximum number of thinking tokens per response.
            Must be less than maximum output tokens. Only applies to Anthropic models.
        """,
    )

    # OpenAI settings
    openai_reasoning_effort: ReasoningEffort = Field(
        default="minimal",
        description="Reasoning effort for the model. Only applies to OpenAI models.",
    )
