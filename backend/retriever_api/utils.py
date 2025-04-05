__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from chromadb import Collection
from typing import List
import streamlit as st
from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore

from omegaconf import OmegaConf
config = OmegaConf.load("backend/config/config.yaml")

class Utils:

    @staticmethod
    def load_collection_from_local(name: str = config.collection.name) -> Collection:
        client = chromadb.HttpClient(
            host="http://127.0.0.1:8000",
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=st.secrets["chroma_server_auth_credentials"]
            )
        )
        client.heartbeat()
        return client.get_collection(name=name)


class DocStoreManager:
    """Helper methods for DocStore creation and persistance."""

    @staticmethod
    def create_docs_from_collection(collection: Collection, batch_size: int = 64, light: bool = False) -> List[Document]:
        docs = []
        db_size = collection.count()
        for i in tqdm(range(0, db_size, batch_size), desc="Loading Docs"):
            batch = collection.get(
                include=["metadatas", "documents"],
                limit=batch_size,
                offset=i
            )
            for j in range(len(batch["documents"])):
                if light:
                    metadata={"id": batch["metadatas"][j]["id"]}
                else:
                    metadata={
                        "id": batch["metadatas"][j]["id"],
                        "link": batch["metadatas"][j]["link"],
                        "title": batch["metadatas"][j]["title"],
                        "children": batch["metadatas"][j]["children"],
                    }
                doc = Document(
                    metadata=metadata,
                    text=batch["documents"][j]
                )
                docs.append(doc)
        return docs

    @staticmethod
    def create_nodes(collection: Collection) -> List[BaseNode]:
        node_parser = SimpleNodeParser().from_defaults(
            chunk_size=1e9,
            chunk_overlap=0,
            include_metadata=True
        )
        return node_parser.get_nodes_from_documents(
            DocStoreManager.create_docs_from_collection(collection), 
            show_progress=True
        )

    @staticmethod
    def persist_nodes_to_docstore(nodes: List[BaseNode], persist_path: str) -> None:
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        docstore.persist(
            persist_path=persist_path
        )

    @staticmethod
    def load_docstore(persist_path: str):
        return SimpleDocumentStore.from_persist_path(
            persist_path=persist_path
        )