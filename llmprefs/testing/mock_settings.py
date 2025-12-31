from pathlib import Path

from llmprefs.api.structs import LLM
from llmprefs.settings import Settings


class MockSettings(Settings, cli_parse_args=False):
    input_path: Path = Path("in.csv")
    output_path: Path = Path("out.jsonl")
    model: LLM = LLM.MOCK_MODEL
    parsing_model: LLM = LLM.MOCK_MODEL
