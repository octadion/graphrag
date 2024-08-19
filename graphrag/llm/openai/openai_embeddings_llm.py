# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The EmbeddingsLLM class."""

from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)

from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes


class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        extra_body = getattr(self.configuration, 'extra_body', None)
        extra_body_from_kwargs = kwargs.pop('extra_body', None)
        if extra_body_from_kwargs:
            if extra_body:
                extra_body.update(extra_body_from_kwargs)
            else:
                extra_body = extra_body_from_kwargs
        args = {
            "model": self.configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        if extra_body:
            args['extra_body'] = extra_body
        embedding = await self.client.embeddings.create(
            input=input,
            **args,
        )
        return [d.embedding for d in embedding.data]
