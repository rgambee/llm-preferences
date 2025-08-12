from openai import AsyncOpenAI
from openai.types.shared_params.reasoning import Reasoning

from llmprefs.api.base import BaseApi
from llmprefs.structs import OpenAiApiParams


class OpenAiApi(BaseApi):
    def __init__(
        self,
        client: AsyncOpenAI,
        params: OpenAiApiParams,
    ) -> None:
        self._client = client
        self._params = params

    @property
    def params(self) -> OpenAiApiParams:
        return self._params

    async def submit(
        self,
        prompt: str,
    ) -> str:
        response = await self._client.responses.create(
            input=prompt,
            model=self._params.model.value,
            max_output_tokens=self._params.max_tokens,
            instructions=self._params.system_prompt,
            temperature=self._params.temperature,
            reasoning=Reasoning(
                effort=self._params.reasoning_effort,
            ),
        )
        return response.output_text
