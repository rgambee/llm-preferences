from abc import ABC, abstractmethod

from pydantic import BaseModel

from llmprefs.structs import LLM


class ApiParameters(BaseModel):
    model: LLM
    max_tokens: int
    system_prompt: str
    temperature: float


class BaseApi(ABC):
    @abstractmethod
    async def submit(self, prompt: str) -> str:
        pass
