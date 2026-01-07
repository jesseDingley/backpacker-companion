from llama_index.retrievers.bm25 import BM25Retriever
from utils.utils import Utils, DocStoreManager
import Stemmer
import os
from collections import defaultdict
from llama_index.core.schema import QueryBundle
from typing import List, Dict, Any, Tuple
from chromadb import Collection

import logging
logging.basicConfig(
    format="%(levelname)s:  %(message)s"
)

class VectorRetriever:
    """Vector Retriever using Chroma collection."""
    
    def __init__(
            self, 
            collection: Collection, 
            top_k: int, 
            threshold: float
        ) -> None:
        self.collection = collection
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieves relevant documents (id, score, text, metadata) from Chroma collection.
        Results are filtered and sorted.
        
        Args:
            query (str): query string

        Returns:
            List[Dict[str, Any]]: sorted list of retrieved documents in dict format
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k
        )
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "id": results["metadatas"][0][i]["id"],
                    "score": results["distances"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )
        
        return sorted(
            [elt for elt in formatted_results if elt["score"] <= self.threshold],
            key=lambda x: x["score"], 
            reverse=False
        )


class CustomBM25Retriever(BM25Retriever):
    """BM25 Retriver from Llamaindex."""

    @classmethod
    def from_defaults(
        cls,
        docstore_path = None,
        stemmer = None,
        language = "en",
        similarity_top_k = ...,
        verbose = False,
        skip_stemming = False,
        token_pattern = r"(?u)\b\w\w+\b",
        tokenizer = None,
        threshold = 0.0
    ):
        assert isinstance(docstore_path, str)
        assert os.path.exists(docstore_path), "docstore does not exist"

        docstore = DocStoreManager.load_docstore(docstore_path)

        instance = super().from_defaults(
            None, 
            None, 
            docstore, 
            stemmer, 
            language, 
            similarity_top_k, 
            verbose, 
            skip_stemming, 
            token_pattern, 
            tokenizer
        )

        instance.threshold = threshold

        return instance


    def retrieve(self, query_bundle: QueryBundle) -> List[Dict[str, Any]]:
        """
        Retrieves relevant documents (id, score, text, metadata) from docstore index.
        Results are filtered and sorted.
        
        Args:
            query (QueryBundle): query string

        Returns:
            List[Dict[str, Any]]: sorted list of retrieved documents in dict format
        """
        raw_nodes = super().retrieve(query_bundle)
        formatted_results = []
        for node in raw_nodes:
            formatted_results.append(
                {
                    "id": node.metadata["id"],
                    "score": node.score,
                    "text": node.text,
                    "metadata": node.metadata
                }
            )
        return sorted(
            [elt for elt in formatted_results if elt["score"] >= self.threshold],
            key=lambda x: x["score"], 
            reverse=True
        )


class HybridRetriever:
    """Hybrid Retriver."""

    def __init__(
        self, 
        path_docstore: str,
        pre_top_k: int = 50, 
        bm25_threshold: float = 6.0,
        vector_threshold: float = 0.8,
        w_bm25: float = 0.5,
        w_vector: float = 0.5,
        rrf_k: int = 60,
        max_docs_plus_children: int = 15,
    ):
        
        chroma_client = Utils.init_chroma_client()
        self.collection = Utils.load_collection_from_client(chroma_client)
        
        self.bm25_retriever = CustomBM25Retriever.from_defaults(
            docstore_path=path_docstore,
            similarity_top_k=pre_top_k,
            stemmer=Stemmer.Stemmer("english"),
            language="english",
            threshold=bm25_threshold,
        )

        self.vector_retriever = VectorRetriever(
            collection=self.collection,
            top_k=pre_top_k,
            threshold=vector_threshold
        )

        self.rrf_k = rrf_k
        self.default_rank = self.rrf_k + 1

        self.w_bm25 = w_bm25
        self.w_vector = w_vector

        self.max_docs_plus_children = max_docs_plus_children

        logging.warning("Hybrid Retriever init complete.")

    def refresh_token(self):
        """
        Refresh Chroma Service Google ID Token 
        """
        chroma_client = Utils.init_chroma_client()
        self.collection = Utils.load_collection_from_client(chroma_client)
        self.vector_retriever.collection = self.collection

    @staticmethod
    def get_documents_ranking(retriever_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Gets documents ranking (id -> rank) from a list of retrieved documents.

        Args:
            retriever_results (List[Dict[str, Any]]): docs from either BM25 or Chroma

        Returns:
            Dict[str, int]: Doc ID to rank mapping
        """
        return {
            doc["id"]: i + 1 for i, doc in enumerate(retriever_results)
        }

    @staticmethod
    def get_id2doc_mapping(
        docs_1: List[Dict[str, Any]], 
        docs_2: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Gets ID -> Document mapping from all unique retrieved documents.

        Args:
            docs_1 (List[Dict[str, Any]]): retrieved docs from 1st retriever 
            docs_2 (List[Dict[str, Any]]): retrieved docs from 2nd retriever 

        Returns:
            Dict[str, Dict[str, Any]]: ID -> Document mapping.
        """
        id2doc_mapping = {}
        docs_concat = docs_1 + docs_2
        for doc in docs_concat:
            if not doc["id"] in id2doc_mapping:
                id2doc_mapping[doc["id"]] = {
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                }
        return id2doc_mapping

    @staticmethod
    def sort_and_filter_rrf_scores(rrf_scores: Dict[str, float], k: int, threshold: float) -> List[Tuple[str, float]]:
        """
        Sorts and filters RRF scores.

        Args:
            rrf_scores (Dict[str, float]): ID -> RRF Score mapping
            k (int): num of top docs to return
            threshold (float): minimum score a doc must have.

        Returns:
            List[Tuple[str, float]]: Sorted Filtered List of (ID, score) pairs.
        """
        rrf_scores = {
            k: score for k, score in rrf_scores.items() if score >= threshold
        }

        rrf_scores = sorted(
            rrf_scores.items(), 
            key = lambda x: x[1], 
            reverse=True
        )
        return rrf_scores[:k]

    def fill_missing_ranks(
        self, 
        ranking_1:  Dict[str, int], 
        ranking_2:  Dict[str, int]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Fills missing ranks 
        i.e. if a document from one ranking is missing in the other we'll assign in rank k+1.

        Args:
            ranking_1 (Dict[str, int]): ranking from 1st retriever
            ranking_2 (Dict[str, int]): ranking from 2nd retriever

        Returns:
            Tuple[Dict[str, int], Dict[str, int]]: updated rankings
        """

        ranking_1_docs = set(ranking_1.keys())
        ranking_2_docs = set(ranking_2.keys())

        all_retrieved_docs = ranking_1_docs.union(ranking_2_docs)

        for doc_id in all_retrieved_docs:
            if doc_id not in ranking_1:
                ranking_1[doc_id] = self.default_rank
            if doc_id not in ranking_2:
                ranking_2[doc_id] = self.default_rank

        return ranking_1, ranking_2

    def compute_rrf_scores(
        self, 
        ranking_bm25: Dict[str, int], 
        ranking_vector: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Computes RRF Scores given pair of rankings.

        Args:
            ranking_bm25 (Dict[str, int]): ID -> rank mapping for BM25 results
            ranking_vector (Dict[str, int]): ID -> rank mapping for Vector results

        Returns:
            Dict[str, float]: ID -> RRF Score mapping
        """
        
        rrf_scores = defaultdict(float)

        for doc_id, rank in ranking_bm25.items():
            rrf_scores[doc_id] += self.w_bm25 * (1 / (self.rrf_k + rank))

        for doc_id, rank in ranking_vector.items():
            rrf_scores[doc_id] += self.w_vector * (1 / (self.rrf_k + rank))

        return rrf_scores

    def get_children(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For a given set of documents, 
        returns original docs and associated children.
        """

        counter = 0

        docs_with_children = []

        for i, doc in enumerate(docs):

            docs_with_children.append(doc)
            counter += 1

            if i < 5: # only top n documents can be considered eligible for child retrieval.
                doc_child_ids = doc["metadata"]["children"].split("[SEP]")
            else:
                doc_child_ids = [""]

            if doc_child_ids != [""]:

                doc_children = self.collection.get(
                    where={
                        "id": {
                            "$in": doc_child_ids
                        }
                    },
                    include=['metadatas', 'documents']
                )

                formatted_children = []
                for i in range(len(doc_children["ids"])):
                    formatted_children.append(
                        {
                            "id": doc_children["metadatas"][i]["id"],
                            "text": doc_children["documents"][i],
                            "metadata": doc_children["metadatas"][i],
                        }
                    )
                
                formatted_children = sorted(
                    formatted_children, 
                    key=lambda x: doc_child_ids.index(x["id"])
                )

                docs_with_children += formatted_children
                counter += len(formatted_children)

            if counter >= self.max_docs_plus_children:
                return docs_with_children[:self.max_docs_plus_children]

        return docs_with_children

    def retrieve(self, query: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Retrieval given query.

        Args:
            query (str): query string
            k (str): num of top docs to return

        Returns:
            List[Dict[str, Any]]: retrieved documents.
        """

        bm25_docs = self.bm25_retriever.retrieve(query)
        vector_docs = self.vector_retriever.retrieve(query)

        bm25_ranks = HybridRetriever.get_documents_ranking(bm25_docs)
        vector_ranks = HybridRetriever.get_documents_ranking(vector_docs)

        bm25_ranks, vector_ranks = self.fill_missing_ranks(
            bm25_ranks, 
            vector_ranks
        )

        rrf_scores = self.compute_rrf_scores(
            bm25_ranks, 
            vector_ranks
        )

        rrf_scores = self.sort_and_filter_rrf_scores(rrf_scores, k, threshold)
        
        id2doc_mapping = HybridRetriever.get_id2doc_mapping(
            bm25_docs, 
            vector_docs
        )

        retrieved_docs = []
        for doc_id, score in rrf_scores:
            doc_to_return = id2doc_mapping[doc_id]
            doc_to_return["metadata"]["score"] = score
            retrieved_docs.append(doc_to_return)

        retrieved_docs_plus_children = self.get_children(retrieved_docs)
            
        return retrieved_docs_plus_children