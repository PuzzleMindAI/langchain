"""Wrappers on top of vector stores."""
from langchain.vectorstores.analyticdb import AnalyticDB
from langchain.vectorstores.annoy import Annoy
from langchain.vectorstores.atlas import AtlasDB
from langchain.vectorstores.awadb import AwaDB
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.clickhouse import Clickhouse, ClickhouseSettings
from langchain.vectorstores.deeplake import DeepLake
from langchain.vectorstores.docarray import DocArrayHnswSearch, DocArrayInMemorySearch
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.lancedb import LanceDB
from langchain.vectorstores.matching_engine import MatchingEngine
from langchain.vectorstores.milvus import Milvus
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain.vectorstores.myscale import MyScale, MyScaleSettings
from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain.vectorstores.pinecone import Pinecone
from langchain.vectorstores.qdrant import Qdrant
from langchain.vectorstores.redis import Redis
from langchain.vectorstores.singlestoredb import SingleStoreDB
from langchain.vectorstores.sklearn import SKLearnVectorStore
from langchain.vectorstores.supabase import SupabaseVectorStore
from langchain.vectorstores.tair import Tair
from langchain.vectorstores.tigris import Tigris
from langchain.vectorstores.typesense import Typesense
from langchain.vectorstores.vectara import Vectara
from langchain.vectorstores.weaviate import Weaviate
from langchain.vectorstores.zilliz import Zilliz
from langchain.vectorstores.azuresearch import AzureSearch

__all__ = [
    "AzureSearch",
    "Redis",
    "ElasticVectorSearch",
    "FAISS",
    "VectorStore",
    "Pinecone",
    "Weaviate",
    "Qdrant",
    "Milvus",
    "Zilliz",
    "SingleStoreDB",
    "Chroma",
    "OpenSearchVectorSearch",
    "AtlasDB",
    "DeepLake",
    "Annoy",
    "MongoDBAtlasVectorSearch",
    "MyScale",
    "MyScaleSettings",
    "SKLearnVectorStore",
    "SupabaseVectorStore",
    "AnalyticDB",
    "Vectara",
    "Tair",
    "LanceDB",
    "DocArrayHnswSearch",
    "DocArrayInMemorySearch",
    "Typesense",
    "Clickhouse",
    "ClickhouseSettings",
    "Tigris",
    "MatchingEngine",
    "AwaDB",
]
