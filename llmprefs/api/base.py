from abc import ABC, abstractmethod


class BaseApi(ABC):
    @abstractmethod
    async def submit(self, prompt: str) -> str:
        pass
