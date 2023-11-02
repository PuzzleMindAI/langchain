"""
Test of Astra DB vector store class `AstraDB`

Required to run this test:
    - a recent `astrapy` Python package available
    - an Astra DB instance;
    - the two environment variables set:
        export ASTRA_DB_API_ENDPOINT="https://<DB-ID>-us-east1.apps.astra.datastax.com"
        export ASTRA_DB_APPLICATION_TOKEN="AstraCS:........."
    - optionally this as well (otherwise defaults are used):
        export ASTRA_DB_KEYSPACE="my_keyspace"
"""

import os
import json
import math
from typing import List

import pytest

from langchain.vectorstores import AstraDB
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

# Ad-hoc embedding classes:

class SomeEmbeddings(Embeddings):
    """
    Turn a sentence into an embedding vector in some way.
    Not important how. It is deterministic is all that counts.
    """

    def __init__(self, dimension):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(txt) for txt in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        unnormed0 = [ord(c) for c in text[: self.dimension]]
        unnormed = (unnormed0 + [1] + [0] * (self.dimension - 1 - len(unnormed0)))[
            : self.dimension
        ]
        norm = sum(x * x for x in unnormed) ** 0.5
        normed = [x / norm for x in unnormed]
        return normed

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


class ParserEmbeddings(Embeddings):
    """
    Parse input texts: if they are json for a List[float], fine.
    Otherwise, return all zeros and call it a day.
    """

    def __init__(self, dimension):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(txt) for txt in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            vals = json.loads(text)
            assert len(vals) == self.dimension
            return vals
        except Exception:
            print(f'[ParserEmbeddings] Returning a moot vector for "{text}"')
            return [0.0] * self.dimension

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def _can_use_astradb() -> bool:
    try:
        import astrapy
        _ = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
        _ = os.environ["ASTRA_DB_API_ENDPOINT"]
        return True
    except Exception:
        return False


@pytest.fixture(scope="function")
def store_someemb():
    emb = SomeEmbeddings(dimension=2)
    v_store = AstraDB(
        embedding=emb,
        collection_name="lc_test_s",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )
    yield v_store
    v_store.delete_collection()


@pytest.fixture(scope="function")
def store_parseremb():
    emb = ParserEmbeddings(dimension=2)
    v_store = AstraDB(
        embedding=emb,
        collection_name="lc_test_s",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )
    yield v_store
    v_store.delete_collection()


@pytest.mark.skipif(not _can_use_astradb(), reason="Missing astrapy or envvars")
class TestAstraDB:

    def test_astradb_vectorstore_create_delete(self):
        """Create and delete."""
        emb = SomeEmbeddings(dimension=2)
        # creation by passing the connection secrets
        v_store = AstraDB(
            embedding=emb,
            collection_name="lc_test_1",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        v_store.delete_collection()
        # Creation by passing a ready-made astrapy client:
        from astrapy.db import AstraDB as LibAstraDB
        astra_db_client = LibAstraDB(
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        v_store_2 = AstraDB(
            embedding=emb,
            collection_name="lc_test_2",
            astra_db_client=astra_db_client,
        )
        v_store_2.delete_collection()

    def test_astradb_vectorstore_from_x(self):
        """from_texts and from_documents methods."""
        emb = SomeEmbeddings(dimension=2)
        # from_texts
        v_store = AstraDB.from_texts(
            texts=["Hi", "Ho"],
            embedding=emb,
            collection_name="lc_test_f",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        assert v_store.similarity_search("Ho", k=1)[0].page_content == "Ho"
        v_store.clear()

        # from_texts
        v_store_2 = AstraDB.from_documents(
            [
                Document(page_content="Hee"),
                Document(page_content="Haa"),
            ],
            embedding=emb,
            collection_name="lc_test_f",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        assert v_store_2.similarity_search("Haa", k=1)[0].page_content == "Haa"
        v_store_2.clear()
        # manual collection delete
        v_store_2.delete_collection()

    def test_astradb_vectorstore_crud(self, store_someemb):
        """Basic add/delete/update behaviour."""
        res0 = store_someemb.similarity_search("Abc", k=2)
        assert res0 == []
        # write and check again
        store_someemb.add_texts(
            texts=["aa", "bb", "cc"],
            metadatas=[
                {"k": "a", "ord": 0},
                {"k": "b", "ord": 1},
                {"k": "c", "ord": 2},
            ],
            ids=["a", "b", "c"],
        )
        res1 = store_someemb.similarity_search("Abc", k=5)
        assert {doc.page_content for doc in res1} == {"aa", "bb", "cc"}
        # partial overwrite and count total entries
        store_someemb.add_texts(
            texts=["cc", "dd"],
            metadatas=[
                {"k": "c_new", "ord": 102},
                {"k": "d_new", "ord": 103},
            ],
            ids=["c", "d"],
        )
        res2 = store_someemb.similarity_search("Abc", k=10)
        assert len(res2) == 4
        # pick one that was just updated and check its metadata
        res3 = store_someemb.similarity_search_with_score_id("cc", k=1)
        doc3, score3, id3 = res3[0]
        assert doc3.page_content == "cc"
        assert doc3.metadata == {"k": "c_new", "ord": 102}
        assert score3 > 0.999  # leaving some leeway for approximations...
        assert id3 == "c"
        # delete and count again
        del1_res = store_someemb.delete(["b"])
        assert del1_res == True
        del2_res = store_someemb.delete(["a", "c", "Z!"])
        assert del2_res == False  # a non-existing ID was supplied
        assert len(store_someemb.similarity_search("xy", k=10)) == 1
        # clear store
        store_someemb.clear()
        assert store_someemb.similarity_search("Abc", k=2) == []
        # add_documents with "ids" arg passthrough
        store_someemb.add_documents(
            [
                Document(page_content="vv", metadata={"k": "v", "ord": 204}),
                Document(page_content="ww", metadata={"k": "w", "ord": 205}),
            ],
            ids=["v", "w"],
        )
        assert len(store_someemb.similarity_search("xy", k=10)) == 2
        res4 = store_someemb.similarity_search("ww", k=1)
        assert res4[0].metadata["ord"] == 205

    def test_astradb_vectorstore_mmr(self, store_parseremb):
        """
        MMR testing. We work on the unit circle with angle multiples
        of 2*pi/20 and prepare a store with known vectors for a controlled
        MMR outcome.
        """
        def _v_from_i(i, N):
            angle = 2 * math.pi * i / N
            vector = [math.cos(angle), math.sin(angle)]
            return json.dumps(vector)

        i_vals = [0, 4, 5, 13]
        N_val = 20
        store_parseremb.add_texts(
            [_v_from_i(i, N_val) for i in i_vals],
            metadatas=[{"i": i} for i in i_vals]
        )
        res1 = store_parseremb.max_marginal_relevance_search(
            _v_from_i(3, N_val),
            k=2,
            fetch_k=3,
        )
        res_i_vals = {doc.metadata["i"] for doc in res1}
        assert res_i_vals == {0, 4}

    def test_astradb_vectorstore_metadata(self, store_someemb):
        """Metadata filtering."""
        store_someemb.add_documents([
            Document(
                page_content="q",
                metadata={"ord": ord("q"), "group": "consonant"},
            ),
            Document(
                page_content="w",
                metadata={"ord": ord("w"), "group": "consonant"},
            ),
            Document(
                page_content="r",
                metadata={"ord": ord("r"), "group": "consonant"},
            ),
            Document(
                page_content="e",
                metadata={"ord": ord("e"), "group": "vowel"},
            ),
            Document(
                page_content="i",
                metadata={"ord": ord("i"), "group": "vowel"},
            ),
            Document(
                page_content="o",
                metadata={"ord": ord("o"), "group": "vowel"},
            ),
        ])
        # no filters
        res0 = store_someemb.similarity_search("x", k=10)
        assert {doc.page_content for doc in res0} == set("qwreio")
        # single filter
        res1 = store_someemb.similarity_search(
            "x",
            k=10,
            filter={"group": "vowel"},
        )
        assert {doc.page_content for doc in res1} == set("eio")
        # multiple filters
        res2 = store_someemb.similarity_search(
            "x",
            k=10,
            filter={"group": "consonant", "ord": ord("q")},
        )
        assert {doc.page_content for doc in res2} == set("q")
        # excessive filters
        res3 = store_someemb.similarity_search(
            "x",
            k=10,
            filter={"group": "consonant", "ord": ord("q"), "case": "upper"},
        )
        assert res3 == []

    def test_astradb_vectorstore_similarity_scale(self, store_parseremb):
        """Scale of the similarity scores."""
        store_parseremb.add_texts(
            texts=[
                json.dumps([1, 1]),
                json.dumps([-1, -1]),
            ],
            ids=["near", "far"],
        )
        res1 = store_parseremb.similarity_search_with_score(
            json.dumps([0.5, 0.5]),
            k=2,
        )
        scores = [sco for _, sco in res1]
        sco_near, sco_far = scores
        assert abs(1 - sco_near) < 0.001 and abs(sco_far) < 0.001
