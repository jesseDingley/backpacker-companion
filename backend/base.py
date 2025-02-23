from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import streamlit as st
import logging
from omegaconf import OmegaConf


class Base:
    """
    Base class that defines useful attributes for the repo

    Attributes
    ------------
    path_post_urls (str): Path to txt file containing post urls
    path_title_image (str): Path to title image (page logo)
    path_assistant_icon (str): Path to assistant icon (in chat)
    path_sidebar_mid (str): Path to sidebar markdown file
    embeddings (HuggingFaceEmbeddings): Embedding function for vectorDB
    chroma_client (ClientAPI): Chroma client connecting to gcloud vm
    
    """

    def __init__(self) -> None:

        self.config = OmegaConf.load("backend/config/config.yaml")

        self.NAME = self.config.app.name
        self.LLM = self.config.app.llm
        self.debug = self.config.app.debug

        self.collection_config = {
            "NAME": self.config.collection.name,
            "BASE_URL": self.config.collection.base_url,
            "USER_AGENT": self.config.collection.user_agent,
            "EMBEDDING_MODEL": self.config.collection.embedding_model
        }

        self.paths = {
            "POST_URLS": self.config.paths.data.urls,
            "TITLE_IMAGE": self.config.paths.images.title_image,
            "ASSISTANT_ICON": self.config.paths.images.assistant_icon,
            "SIDEBAR": self.config.paths.ui.sidebar
        }
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.collection_config["EMBEDDING_MODEL"]
        )

        self.chroma_client = chromadb.HttpClient(
            host=st.secrets["chroma_ip"],
            port=8000,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=st.secrets["chroma_server_auth_credentials"]
            )
        )

        assert self.chroma_client.heartbeat() > 0

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )