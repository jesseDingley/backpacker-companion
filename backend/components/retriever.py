from typing import Any
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import chain
from langchain_core.documents import Document
from typing import List


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
        vectordb: Chroma,
        k: int = 4,
        threshold: float | None = None,
        rerank: bool = False,
        rerank_top_k: int = 15,
    ) -> Any:

        instance = super().__new__(cls)

        instance.vectordb = vectordb
        instance.k = k
        instance.threshold = threshold
        instance.rerank = rerank
        instance.rerank_top_k = rerank_top_k

        return instance.initialize_retriever()

    @staticmethod
    def initialize_vectordb_retriever(
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
        def retriever(query: str) -> List[Document]:
            docs, scores = zip(*vectordb.similarity_search_with_score(query, k=k))
            for doc, score in zip(docs, scores):
                doc.metadata["score"] = score

            if threshold is not None:
                return [doc for doc in docs if doc.metadata["score"] <= threshold]
            return docs

        return retriever

    def initialize_retriever(self) -> Any:
        """Initializes retriever."""
        if not self.rerank:
            return self.initialize_standard_retriever()
        return self.initialize_rerank_retriever()

    def initialize_standard_retriever(self) -> VectorStoreRetriever:
        """Initializes standard retriever."""

        return Retriever.initialize_vectordb_retriever(
            vectordb=self.vectordb, k=self.k, threshold=self.threshold
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
