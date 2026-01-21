import os
from dotenv import load_dotenv
import requests
from time import time, sleep
import logging

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_ollama import ChatOllama

import chromadb
import streamlit as st
from omegaconf import OmegaConf
from huggingface_hub import login

from google.oauth2.service_account import IDTokenCredentials
import google.auth.transport.requests

from backend.config.const import CST

@st.cache_resource(show_spinner="Logging into HF")
def login_hf() -> None:
    """Login to HF in order to access inference endpoints."""
    login(token=st.secrets["secrets"]["huggingface_api_key"])

@st.cache_resource(show_spinner="Loading Embeddings")
def load_emb_ft(embedding_model: str) -> HuggingFaceEmbeddings:
    """
    Load HF embedding function.
    
    Args:
        embedding_model (str): embedding model name

    Returns:
        HuggingFaceEmbeddings: embedding function.
    """
    return HuggingFaceEmbeddings(
        model_name=embedding_model
    )

@st.cache_resource(show_spinner="Initializing Chroma Client")
def init_chroma_client() -> chromadb.HttpClient:
    """Returns Chroma HTTP client."""

    credentials = IDTokenCredentials.from_service_account_info(
        st.secrets["secrets"]["service_account_data"], 
        target_audience=st.secrets["secrets"]["chroma_endpoint"]
    )

    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    chroma_client = chromadb.HttpClient(
        host=st.secrets["secrets"]["chroma_endpoint"],
        port=443,
        ssl=True,
        headers={
            "Authorization": f"Bearer {credentials.token}"
        }
    )

    assert chroma_client.heartbeat() > 0
    return chroma_client

@st.cache_resource(show_spinner="Initializing LLM")
def init_llm(llm: str) -> HuggingFaceEndpoint | ChatOllama:
    """
    Inits LLM endpoint. 
    If running in production -> HF endpoint
    If running locally       -> Ollama
    
    Args:
        llm (str): can be repo id or endpoint url.

    Returns:
        HuggingFaceEndpoint | ChatOllama
    """
    if llm.startswith("http"):
        return HuggingFaceEndpoint(
            endpoint_url=llm,
            task="text-generation",
            max_new_tokens=CST.MAX_NEW_TOKENS,
            temperature=CST.TEMPERATURE,
            top_k=CST.TOP_K,
            top_p=CST.TOP_P,
            repetition_penalty=CST.REPEAT_PENALTY,
            callbacks=[StreamingStdOutCallbackHandler()],
            streaming=True,
            stop_sequences=[
                "<unk>", "</s>", "Assistant", "User"
            ],
        )
            
    return ChatOllama(
        model="mistral:latest",
        temperature=CST.TEMPERATURE,
        top_k=CST.TOP_K,
        top_p=CST.TOP_P,
        repeat_penalty=CST.REPEAT_PENALTY,
        callbacks=[StreamingStdOutCallbackHandler()],
        stop=[
            "<unk>", "</s>", "Assistant", "User"
        ],
    )

@st.cache_resource(show_spinner="Waking up retriever server")
def wake_up_server() -> None:
    """Wakes up retriever cloud run service."""

    t0 = time()

    while True:

        response = requests.get(
            st.secrets["secrets"]["retriever_endpoint"] + "/"
        )

        if response.status_code == 200:
            return
        
        if time() - t0 > 120: # waited more than two minutes
            raise requests.RequestException("Server startup failed.")
        
        sleep(5)

@st.cache_resource(show_spinner="Warming up LLM endpoint.")
def wake_up_llm_endpoint() -> None:
    """Wakes up HF LLM endpoint."""

    while True:

        response = requests.get(
            st.secrets["secrets"]["llm_endpoint"],
            headers={
                "Authorization": f"Bearer {st.secrets['secrets']['huggingface_api_key']}"
            }
        )

        if response.status_code in [200, 404]:
            return

        sleep(5)

class Base:
    """
    Base class that defines useful attributes for the repo and runs pre-chat functions.

    Attributes
    ------------
    config (OmegaConf): Project config
    NAME (str): Name of assistant
    LLM_ENDPOINT (str): name of LLM Endpoint
    RETRIEVER (str): Type of retriever (hybrid usually)
    debug (bool): True if we want debug logs
    collection_config (dict): Config for Chroma collection
    paths (dict): Dict of paths to various utilities (images etc)
    embeddings (HuggingFaceEmbeddings): Emb function for creating chroma collection
    chroma_client (chromadb.HttpClient): Chroma client.
    llm (HuggingFaceEndpoint | ChatOllama): LLM endpoint (in prod: HF API / local: Ollama)
    """

    def __init__(self) -> None:

        load_dotenv()
        login_hf()

        self.config = OmegaConf.load("backend/config/config.yaml")

        self.NAME = self.config.app.name
        self.LLM_ENDPOINT = self.config.app.llm if os.environ.get("ENV") == "dev" else st.secrets["secrets"]["llm_endpoint"]
        self.RETRIEVER = self.config.app.retriever
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

        if os.environ.get("ENV") == "dev":
            self.embeddings = load_emb_ft(self.collection_config["EMBEDDING_MODEL"])
            self.chroma_client = init_chroma_client()

        self.llm = init_llm(llm=self.LLM_ENDPOINT)

        wake_up_server()

        if os.environ.get("ENV") != "dev":
            wake_up_llm_endpoint()

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )