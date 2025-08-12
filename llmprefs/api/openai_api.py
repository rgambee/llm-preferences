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
        self.client = client
        self.params = params

    async def submit(
        self,
        prompt: str,
    ) -> str:
        response = await self.client.responses.create(
            input=prompt,
            model=self.params.model.value,
            max_output_tokens=self.params.max_tokens,
            instructions=self.params.system_prompt,
            temperature=self.params.temperature,
            reasoning=Reasoning(
                effort=self.params.reasoning_effort,
            ),
        )
        return response.output_text
