from abc import ABC, abstractmethod

from llmprefs.api.structs import AnyApiParameters


class BaseApi(ABC):
    @property
    @abstractmethod
    def params(self) -> AnyApiParameters:
        pass

    @abstractmethod
    async def submit(self, prompt: str) -> str:
        pass
