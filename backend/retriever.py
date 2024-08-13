from typing import Any
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.vectorstores import VectorStoreRetriever


class Retriever:
    """Defines different retrievers."""

    def __new__(
        cls,
        vector_db: Chroma,
        search_type: str = "similarity",
        k: int = 3,
        score_threshold: float | None = None,
        rerank: bool = False,
        rerank_top_n: int = 4,
        rerank_top_k: int = 15,
    ) -> Any:

        instance = super().__new__(cls)

        instance.vector_db = vector_db
        instance.search_type = search_type
        instance.k = k
        instance.score_threshold = score_threshold
        instance.rerank = rerank
        instance.rerank_top_n = rerank_top_n
        instance.rerank_top_k = rerank_top_k

        return instance.initialize_retriever()

    def initialize_retriever(self) -> Any:
        """Init retriever."""
        if not self.rerank:
            return self.initialize_standard_retriever()
        return self.initialize_rerank_retriever()

    def initialize_standard_retriever(self) -> VectorStoreRetriever:
        """Init standard retriever."""

        vectordb_retriever_kwargs = {
            "search_type": self.search_type,
            "search_kwargs": {
                "k": self.k,
            },
        }

        if self.search_type == "similarity_score_threshold":
            vectordb_retriever_kwargs["search_kwargs"][
                "score_threshold"
            ] = self.score_threshold

        return self.vector_db.as_retriever(**vectordb_retriever_kwargs)

    def initialize_rerank_retriever(self) -> ContextualCompressionRetriever:
        """Init reranker retriever."""

        vectordb_retriever_kwargs_pre_rerank = {
            "search_kwargs": {
                "k": self.rerank_top_k,
            }
        }

        vectordb_retriever = self.vector_db.as_retriever(
            **vectordb_retriever_kwargs_pre_rerank
        )

        compressor = FlashrankRerank(top_n=self.rerank_top_n)

        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=vectordb_retriever
        )
