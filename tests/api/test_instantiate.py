import pytest

from llmprefs.api.instantiate import LLM_TO_PROVIDER, instantiate_api
from llmprefs.api.structs import LLM
from llmprefs.testing.mock_settings import MockSettings


class TestLlmToProvider:
    def test_all_models_present(self) -> None:
        for llm in LLM:
            assert llm in LLM_TO_PROVIDER


class TestInstantiateApi:
    @pytest.mark.parametrize("model", LLM)
    def test_instantiate_api(self, model: LLM) -> None:
        settings = MockSettings(model=model)
        instantiate_api(settings)
