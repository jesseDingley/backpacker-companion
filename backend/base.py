from backend.const import CST
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import streamlit as st
import os
import logging


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

        self.path_post_urls = os.path.join(CST.PATH_DATA, f"{CST.URLS}.txt")
        self.path_title_image = os.path.join(CST.PATH_IMAGES, f"{CST.TITLE_IMAGE}.png")
        self.path_assistant_icon = os.path.join(
            CST.PATH_IMAGES, f"{CST.ASSISTANT_ICON}.png"
        )
        self.path_sidebar_md = os.path.join(CST.PATH_UI, f"{CST.SIDEBAR}.md")

        self.embeddings = HuggingFaceEmbeddings(model_name=CST.EMBEDDING_MODEL)

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