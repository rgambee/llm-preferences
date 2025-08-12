from abc import ABC, abstractmethod

from llmprefs.structs import AnyApiParams


class BaseApi(ABC):
    @property
    @abstractmethod
    def params(self) -> AnyApiParams:
        pass

    @abstractmethod
    async def submit(self, prompt: str) -> str:
        pass
