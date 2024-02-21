"""Test the public API of the tools package."""
from langchain_community.vectorstores import __all__ as public_api

_EXPECTED = [
    "AlibabaCloudOpenSearch",
    "AlibabaCloudOpenSearchSettings",
    "AnalyticDB",
    "Annoy",
    "ApacheDoris",
    "AtlasDB",
    "AwaDB",
    "AzureSearch",
    "Bagel",
    "Cassandra",
    "AstraDB",
    "Chroma",
    "Clarifai",
    "Clickhouse",
    "ClickhouseSettings",
    "DashVector",
    "DatabricksVectorSearch",
    "DeepLake",
    "Dingo",
    "DocArrayHnswSearch",
    "DocArrayInMemorySearch",
    "ElasticKnnSearch",
    "ElasticVectorSearch",
    "ElasticsearchStore",
    "Epsilla",
    "FAISS",
    "HanaDB",
    "Hologres",
    "KDBAI",
    "LanceDB",
    "Lantern",
    "LLMRails",
    "Marqo",
    "MatchingEngine",
    "Meilisearch",
    "Milvus",
    "MomentoVectorIndex",
    "MongoDBAtlasVectorSearch",
    "MyScale",
    "MyScaleSettings",
    "Neo4jVector",
    "OpenSearchVectorSearch",
    "PGEmbedding",
    "PGVector",
    "Pinecone",
    "Qdrant",
    "Redis",
    "Rockset",
    "SKLearnVectorStore",
    "ScaNN",
    "SemaDB",
    "SingleStoreDB",
    "SQLiteVSS",
    "StarRocks",
    "SupabaseVectorStore",
    "SurrealDBStore",
    "Tair",
    "TileDB",
    "Tigris",
    "TimescaleVector",
    "Typesense",
    "USearch",
    "Vald",
    "Vearch",
    "Vectara",
    "VespaStore",
    "Weaviate",
    "ZepVectorStore",
    "Zilliz",
    "TencentVectorDB",
    "AzureCosmosDBVectorSearch",
    "VectorStore",
    "Yellowbrick",
    "NeuralDBVectorStore",
]


def test_public_api() -> None:
    """Test for regressions or changes in the public API."""
    # Check that the public API is as expected
    assert set(public_api) == set(_EXPECTED)
