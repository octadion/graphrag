# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""OpenAI Embedding model implementation."""

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np
import tiktoken
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.oai.base import OpenAILLMImpl
from graphrag.query.llm.oai.typing import (
    OPENAI_RETRY_ERROR_TYPES,
    OpenaiApiType,
)
from graphrag.query.llm.text_utils import chunk_text
from graphrag.query.progress import StatusReporter
from transformers import AutoTokenizer
from typing import Dict
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import os
import ollama, tiktoken

langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )

MODEL_URI_MAPPING: Dict[str, str] = {
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1",
    "bert-small": "prajjwal1/bert-small",
    "all-MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
    "e5-small": "intfloat/e5-small",
    "bge-small-en": "BAAI/bge-small-en",
    "gte-tiny": "thenlper/gte-tiny",
}

def get_model_uri(model_name: str) -> str:
    return MODEL_URI_MAPPING.get(model_name, model_name)

def count_tokens(provider: str, model_name: str, content: str) -> int:
    try:
        if provider == "openai":
            encoding = tiktoken.encoding_for_model(model_name)
            total_tokens = len(encoding.encode(content))
        elif provider == "ollama":
            model_uri = get_model_uri(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_uri)
            tokens = tokenizer(content)
            total_tokens = len(tokens['input_ids'])
        else:
            raise ValueError('Unsupported provider.')
        return total_tokens
    except Exception as e:
        raise Exception(f"Error counting tokens: {str(e)}")
    
class OpenAIEmbedding(BaseTextEmbedding, OpenAILLMImpl):
    """Wrapper for OpenAI Embedding models."""

    def __init__(
        self,
        api_key: str | None = None,
        azure_ad_token_provider: Callable | None = None,
        model: str = "text-embedding-3-small",
        deployment_name: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        api_type: OpenaiApiType = OpenaiApiType.OpenAI,
        organization: str | None = None,
        encoding_name: str = "cl100k_base",
        max_tokens: int = 8191,
        max_retries: int = 10,
        request_timeout: float = 180.0,
        retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERROR_TYPES,  # type: ignore
        reporter: StatusReporter | None = None,
        provider: str = "openai",
        extra_body: dict | None = None
    ):
        OpenAILLMImpl.__init__(
            self=self,
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            deployment_name=deployment_name,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,  # type: ignore
            organization=organization,
            max_retries=max_retries,
            request_timeout=request_timeout,
            reporter=reporter,
        )

        self.model = model
        self.encoding_name = encoding_name
        self.max_tokens = max_tokens
        self.token_encoder = tiktoken.get_encoding(self.encoding_name)
        self.retry_error_types = retry_error_types
        self.api_base = api_base
        self.provider = provider
        self.extra_body = extra_body

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed text using OpenAI Embedding's sync function.

        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
        Please refer to: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
        """
        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )
        chunk_embeddings = []
        chunk_lens = []
        for chunk in token_chunks:
            try:
                embedding, chunk_len = self._embed_with_retry(chunk, **kwargs)
                chunk_embeddings.append(embedding)
                chunk_lens.append(chunk_len)
            # TODO: catch a more specific exception
            except Exception as e:  # noqa BLE001
                self._reporter.error(
                    message="Error embedding chunk",
                    details={self.__class__.__name__: str(e)},
                )

                continue
        
        if self.provider == "openai":
            chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
            chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
            return chunk_embeddings.tolist()

        elif self.provider == "ollama":
            chunk_embeddings = np.array([chunk['embedding'] for chunk in chunk_embeddings])
            chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
            chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
            return chunk_embeddings.tolist()

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed text using OpenAI Embedding's async function.

        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
        """
        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )
        chunk_embeddings = []
        chunk_lens = []
        embedding_results = await asyncio.gather(*[
            self._aembed_with_retry(chunk, **kwargs) for chunk in token_chunks
        ])
        embedding_results = [result for result in embedding_results if result[0]]
        chunk_embeddings = [result[0] for result in embedding_results]
        chunk_lens = [result[1] for result in embedding_results]
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)  # type: ignore
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        return chunk_embeddings.tolist()

    @observe(as_type='generation')
    def _embed_with_retry(
        self, text: str | tuple, **kwargs: Any
    ) -> tuple[list[float], int]:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    extra_body = kwargs.pop('extra_body', self.extra_body)
                    trace_user_id = extra_body.get('metadata', {}).get('trace_user_id') if extra_body else None
                    if trace_user_id:
                        langfuse_context.update_current_trace(user_id=trace_user_id)
                    kwargs_clone = kwargs.copy()
                    langfuse_context.update_current_observation(
                        input=text,
                        model=self.model,
                        metadata=kwargs_clone
                    )
                    if self.provider == "openai":
                        token_count = count_tokens(self.provider, self.model, text)
                        embedding = (
                            self.sync_client.embeddings.create(
                                input=text,
                                model=self.model,
                                **kwargs,
                            ).data[0].embedding or []
                        )
                        langfuse_context.update_current_observation(
                        usage={
                                "input": token_count,
                                "output": 0,
                                "unit": "TOKENS",
                            }
                        )
                    elif self.provider == "ollama":
                        token_count = count_tokens(self.provider, self.model, text)
                        client = ollama.Client(host=self.api_base)
                        embedding = (
                            client.embeddings(
                                model=self.model,
                                prompt=text) or []
                        )
                        langfuse_context.update_current_observation(
                        usage={
                            "input": token_count,
                            "output": 0,
                            "unit": "TOKENS",
                        }
                    )
                    else:
                        raise ValueError("Unsupported embedding provider")
                    return (embedding, len(text))
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return ([], 0)
        else:
            # TODO: why not just throw in this case?
            return ([], 0)

    async def _aembed_with_retry(
        self, text: str | tuple, **kwargs: Any
    ) -> tuple[list[float], int]:
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            async for attempt in retryer:
                with attempt:
                    if self.provider == "openai":
                        embedding = (
                            await self.async_client.embeddings.create(  # type: ignore
                                input=text,
                                model=self.model,
                                **kwargs,  # type: ignore
                            )
                        ).data[0].embedding or []
                    elif self.provider == "ollama":
                        client = ollama.Client(host=self.api_base)
                        embedding = (
                            client.embeddings(
                                model=self.model,
                                prompt=text) or []
                        )
                    else:
                        raise ValueError("Unsupported embedding provider")
                    return (embedding, len(text))
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return ([], 0)
        else:
            # TODO: why not just throw in this case?
            return ([], 0)
