from openai import AsyncOpenAI
from openai.types.responses import ResponseTextConfigParam
from openai.types.shared_params import Reasoning

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import OpenAiApiParams, OpenAiApiResponse


class OpenAiApi(BaseApi[OpenAiApiResponse]):
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
    ) -> OpenAiApiResponse:
        text: ResponseTextConfigParam = {"format": {"type": "text"}}
        if self._params.tool_config is not None:
            text = {"format": self._params.tool_config}

        raw_reply = await self._client.responses.create(
            input=prompt,
            model=self._params.model.value,
            max_output_tokens=self._params.max_output_tokens,
            instructions=self._params.system_prompt,
            temperature=self._params.temperature,
            reasoning=Reasoning(
                effort=self._params.reasoning_effort,
            ),
            text=text,
        )
        return OpenAiApiResponse.model_validate(
            raw_reply.model_dump(mode="python", by_alias=True)
        )
