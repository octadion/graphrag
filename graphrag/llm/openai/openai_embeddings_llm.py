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
import ollama
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import os, tiktoken
from transformers import AutoTokenizer
from typing import Dict

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
            if isinstance(content, list):
                total_tokens = 0
                for text in content:
                    if not isinstance(text, str):
                        raise ValueError("Each item must be a string.")
                    total_tokens += len(encoding.encode(text))
            elif isinstance(content, str):
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
    
class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration
    @observe(as_type='generation')
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
        embedding_provider = extra_body.get('embedding_provider') if extra_body else None
        embedding_model = extra_body.get('embedding_model') if extra_body else None
        embedding_api_base = extra_body.get('embedding_api_base') if extra_body else None
        
        if extra_body and 'metadata' in extra_body:
            trace_user_id = extra_body['metadata'].get('trace_user_id')
            langfuse_context.update_current_trace(user_id=trace_user_id)

        if embedding_provider == 'ollama':
            client = ollama.Client(host=embedding_api_base)
            embedding_list = []
            for inp in input:
                token_count = count_tokens(embedding_provider, self.configuration.model, inp)
                langfuse_context.update_current_observation(
                    input=inp,
                    model=self.configuration.model,
                    metadata=kwargs.copy(),
                    usage={
                            "input": token_count,
                            "output": 0,
                            "unit": "TOKENS",
                        }
                )
                embedding = client.embeddings(model=embedding_model, prompt=inp)
                embedding_list.append(embedding["embedding"])
            return embedding_list
        
        elif embedding_provider == 'openai':
            token_count = count_tokens(embedding_provider, self.configuration.model, input)
            langfuse_context.update_current_observation(
                input=input,
                model=self.configuration.model,
                metadata=kwargs.copy(),
                usage={
                    "input": token_count,
                    "output": 0,
                    "unit": "TOKENS",
                }
            )
            embedding = await self.client.embeddings.create(
                input=input,
                **args,
            )
            return [d.embedding for d in embedding.data]
        
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
