import os
from unittest.mock import patch

import pytest

from llmprefs.api.instantiate import LLM_TO_PROVIDER, instantiate_api
from llmprefs.api.structs import LLM, ApiStage
from llmprefs.testing.mock_settings import MockSettings


class TestLlmToProvider:
    def test_all_models_present(self) -> None:
        for llm in LLM:
            assert llm in LLM_TO_PROVIDER


class TestInstantiateApi:
    @pytest.mark.parametrize("model", LLM)
    @pytest.mark.parametrize("stage", ApiStage)
    def test_instantiate_api(self, model: LLM, stage: ApiStage) -> None:
        settings = MockSettings(model=model)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            instantiate_api(settings, stage)
