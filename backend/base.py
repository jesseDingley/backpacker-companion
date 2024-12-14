from backend.const import CST
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import streamlit as st
import os


class Base:
    """
    Base class
    """

    def __init__(self) -> None:

        self.path_countries = os.path.join(CST.PATH_DATA, f"{CST.COUNTRIES}.txt")
        self.path_guide_urls = os.path.join(CST.PATH_DATA, f"{CST.URLS}.txt")
        self.path_history = os.path.join(CST.PATH_DATA, f"{CST.HISTORY}.txt")
        self.path_title_image = os.path.join(CST.PATH_IMAGES, f"{CST.TITLE_IMAGE}.png")
        self.path_assistant_icon = os.path.join(
            CST.PATH_IMAGES, f"{CST.ASSISTANT_ICON}.png"
        )
        self.path_sidebar_md = os.path.join(CST.PATH_UI, f"{CST.SIDEBAR}.md")

        self.embeddings = HuggingFaceEmbeddings(model_name=CST.EMBEDDING_MODEL)

        self.chroma_client = chromadb.HttpClient(
            host=CST.CHROMA_IP,
            port=8000,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=st.secrets["chroma_server_auth_credentials"]
            )
        )

        assert self.chroma_client.heartbeat() > 0