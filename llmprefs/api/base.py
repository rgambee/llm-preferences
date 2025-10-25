from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llmprefs.api.structs import AnyApiParameters, AnyApiResponse

ResponseT_co = TypeVar("ResponseT_co", bound=AnyApiResponse, covariant=True)


class BaseApi(ABC, Generic[ResponseT_co]):
    @property
    @abstractmethod
    def params(self) -> AnyApiParameters:
        pass

    @abstractmethod
    async def submit(self, prompt: str) -> ResponseT_co:
        pass
