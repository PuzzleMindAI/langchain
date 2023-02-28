from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Iterable, List, Mapping, Optional

import numpy as np
from redis.commands.search.query import Query

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore


class RediSearch(VectorStore):
    def __init__(
        self, redisearch_url: str, index_name: str, embedding_function: Callable
    ):
        """Initialize with necessary components."""
        try:
            import redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        self.embedding_function = embedding_function
        self.index_name = index_name
        try:
            pool = redis.ConnectionPool.from_url(redisearch_url)
            redis_client = redis.StrictRedis(connection_pool=pool)
        except ValueError as e:
            raise ValueError(f"Your redis connected error: {e}")
        self.client = redis_client

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        # TODO
        prefix = "doc"  # prefix for the document keys

        ids = []
        # Check if index exists
        for i, text in enumerate(texts):
            key = f"{prefix}:{uuid.uuid4().hex}"
            metadata = metadatas[i] if metadatas else {}
            self.client.hset(
                key,
                mapping={
                    "content": text,
                    "content_vector": np.array(
                        self.embedding_function(text), dtype=np.float32
                    ).tobytes(),
                    "metadata": json.dumps(metadata),
                },
            )
            ids.append(key)
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        # Creates embedding vector from user query
        embedding = self.embedding_function(query)

        # Prepare the Query
        return_fields = ["metadata", "content", "vector_score"]
        vector_field = "content_vector"
        hybrid_fields = "*"
        base_query = (
            f"{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]"
        )
        redis_query = (
            Query(base_query)
            .return_fields(*return_fields)
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )
        params_dict: Mapping[str, str] = {
            "vector": str(np.array(embedding).astype(dtype=np.float32).tobytes())
        }

        # perform vector search
        results = self.client.ft(self.index_name).search(redis_query, params_dict)

        documents = [
            Document(page_content=result.content, metadata=json.loads(result.metadata))
            for result in results.docs
        ]

        return documents

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        **kwargs: Any,
    ) -> RediSearch:
        """Construct RediSearch wrapper from raw documents.
        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the RediSearch instance.
            3. Adds the documents to the newly created RediSearch index.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python
                from langchain import RediSearch
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                redisearch = RediSearch.from_texts(
                    texts,
                    embeddings,
                    redisearch_url="redis://username:password@localhost:6379"
                )
        """
        redisearch_url = get_from_dict_or_env(
            kwargs, "redisearch_url", "REDISEARCH_URL"
        )
        try:
            import redis
            from redis.commands.search.field import TextField, VectorField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        try:
            pool = redis.ConnectionPool.from_url(redisearch_url)
            client = redis.StrictRedis(connection_pool=pool)
        except ValueError as e:
            raise ValueError(f"Your redis connected error: {e}")
        embeddings = embedding.embed_documents(texts)
        dim = len(embeddings[0])
        # Constants
        vector_number = len(embeddings)  # initial number of vectors
        # name of the search index if not given
        if not index_name:
            index_name = uuid.uuid4().hex
        prefix = "doc"  # prefix for the document keys
        distance_metric = (
            "COSINE"  # distance metric for the vectors (ex. COSINE, IP, L2)
        )
        content = TextField(name="content")
        metadata = TextField(name="metadata")
        content_embedding = VectorField(
            "content_vector",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": distance_metric,
                "INITIAL_CAP": vector_number,
            },
        )
        fields = [content, metadata, content_embedding]

        # Check if index exists
        try:
            client.ft(index_name).info()
            print("Index already exists")
        except:  # noqa
            # Create RediSearch Index
            client.ft(index_name).create_index(
                fields=fields,
                definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
            )
        for i, text in enumerate(texts):
            key = f"{prefix}:{str(uuid.uuid4().hex)}"
            metadata = metadatas[i] if metadatas else {}
            client.hset(
                key,
                mapping={
                    "content": text,
                    "content_vector": np.array(
                        embeddings[i], dtype=np.float32
                    ).tobytes(),
                    "metadata": json.dumps(metadata),
                },
            )
        return cls(redisearch_url, index_name, embedding.embed_query)
