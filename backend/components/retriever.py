from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import chain
from langchain_core.documents import Document
from typing import List, Literal, Dict, Any
import requests
import streamlit as st
import subprocess

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
        instance.hretriever_endpoint = "http://" + st.secrets["chroma_ip"] + ":8080/retrieve"
        instance.hretriever_endpoint_headers = {
            "Authorization": st.secrets["chroma_server_auth_credentials"]
        }

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
        hretriever_endpoint_headers: dict,
        k: int,
        threshold: float,
        launch_api_on_call: bool = False,
    ) -> VectorStoreRetriever:
        """Initializes hybrid retriever chain."""

        if launch_api_on_call:
            subprocess.run([
                "gcloud", "compute", "ssh", "jessedingley@chroma-instance", 
                "--zone=europe-west1-d", 
                "--command='cd /home/jessedingley/playground && sh launch_api.sh'"
            ])
        
        @chain
        def retriever(fields: Dict[str, Any]) -> List[Document]:

            query = fields["rephrased_input"]

            response = requests.post(
                hretriever_endpoint, 
                json={
                    "query": query,
                    "k": k,
                    "threshold": threshold,
                }, 
                headers=hretriever_endpoint_headers
            )

            assert response.status_code == 200, "API Call Failed."

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
            hretriever_endpoint_headers=self.hretriever_endpoint_headers,
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
