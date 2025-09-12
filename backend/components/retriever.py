from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import chain
from langchain_core.documents import Document
from typing import List, Literal, Dict, Any
import requests
import streamlit as st
import os
from time import time
from requests import HTTPError
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Retriever:
    """Defines different retrievers.

    Attributes
    -----------

    vectordb (Chroma): Chroma Vector DB
    k (int): Max number of documents to return
    threshold (float): Similarity threshold. Any document higher than this will be discarded (only for standard retrieval).
    rerank (bool): Set to True to use reranking
    rerank_top_k (int): Max number of documents to return from vectordb retriever before passing to reranker
    """

    def __new__(
        cls,
        search_type: Literal["vector", "hybrid", "rerank"],
        vectordb: Chroma = None,
        k: int = 4,
        threshold: float = 0.012,
        rerank: bool = False,
        rerank_top_k: int = 15,
    ) -> Any:

        instance = super().__new__(cls)

        instance.search_type = search_type
        instance.vectordb = vectordb
        instance.k = k
        instance.threshold = threshold
        instance.rerank = rerank
        instance.rerank_top_k = rerank_top_k
        instance.hretriever_endpoint = os.path.join(st.secrets["secrets"]["retriever_endpoint"], "retrieve")

        return instance.initialize_retriever()

    @staticmethod
    def initialize_vector_retriever_chain(
        vectordb: Chroma, k: int, threshold: float | None = None
    ) -> VectorStoreRetriever:
        """
        Initializes a vector db retriever with scores.

        Args:
            vectordb (Chroma): Chroma vector db
            k (int): max number of documents to return
            threshold (float): Only documents with a similarity score below threshold will be returned

        Returns:
            VectorStoreRetriever: retriever
        """

        @chain
        def retriever(fields: Dict[str, Any]) -> List[Document]:
            query = fields["rephrased_input"]
            docs, scores = zip(*vectordb.similarity_search_with_score(query, k=k))
            for doc, score in zip(docs, scores):
                doc.metadata["score"] = score

            if threshold is not None:
                return [doc for doc in docs if doc.metadata["score"] <= threshold]
            return docs

        return retriever

    def initialize_vector_retriever(self) -> VectorStoreRetriever:
        """Initializes vector retriever."""

        return Retriever.initialize_vector_retriever_chain(
            vectordb=self.vectordb, k=self.k, threshold=self.threshold
        )

    @staticmethod
    def initialize_hybrid_retriever_chain(
        hretriever_endpoint: str, 
        k: int,
        threshold: float,
        launch_api_on_call: bool = False,
    ) -> VectorStoreRetriever:
        """Initializes hybrid retriever chain."""
     
        @chain
        def retriever(fields: Dict[str, Any]) -> List[Document]:

            query = fields["rephrased_input"]

            t0 = time()

            def call_api():
                return requests.post(
                    hretriever_endpoint, 
                    json={
                        "query": query,
                        "k": k,
                        "threshold": threshold,
                    }, 
                    headers={
                        "Authorization": f"Bearer {fields['jwt_token']}"
                    }
                )
            
            try:
                response = call_api()
                assert response.status_code == 200, f"First attempt failed with {response.status_code}"
            except (AssertionError, requests.exceptions.RequestException):
                try:
                    response = call_api()
                    assert response.status_code == 200,  f"Second attempt failed with {response.status_code}"
                except:
                    if response.status_code == 403:
                        raise HTTPError("403")

            t1 = time()
            logging.info(f"Retrieval time: {t1 - t0:.2f} seconds")

            retrieved_docs = response.json()["res"]

            langchain_docs = []
            for doc in retrieved_docs:
                langchain_docs.append(
                    Document(
                        page_content=doc["text"],
                        metadata=doc["metadata"]
                    )
                )

            return langchain_docs

        return retriever
    
    def initialize_hybrid_retriever(self) -> VectorStoreRetriever:
        """Initializes hybrid retriever."""

        return Retriever.initialize_hybrid_retriever_chain(
            hretriever_endpoint=self.hretriever_endpoint,
            k=self.k,
            threshold=self.threshold
        )

    def initialize_rerank_retriever(self) -> ContextualCompressionRetriever:
        """Init reranker retriever."""

        vectordb_retriever = Retriever.initialize_vectordb_retriever(
            vectordb=self.vectordb,
            k=self.rerank_top_k,
        )

        compressor = FlashrankRerank(top_n=self.k)

        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=vectordb_retriever
        )
    
    def initialize_retriever(self) -> Any:
        """Initializes retriever."""
        if self.search_type == "vector":
            return self.initialize_vector_retriever()
        if self.search_type == "hybrid":
            return self.initialize_hybrid_retriever()
        if self.search_type == "rerank":
            return self.initialize_rerank_retriever()
