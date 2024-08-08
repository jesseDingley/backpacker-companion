from backend.const import CST
from langchain_huggingface import HuggingFaceEmbeddings
import os


class Base:
    """
    Base class
    """

    def __init__(self) -> None:

        self.path_countries = os.path.join(CST.PATH_DATA, f"{CST.COUNTRIES}.txt")
        self.path_guide_urls = os.path.join(CST.PATH_DATA, f"{CST.URLS}.txt")
        self.path_history = os.path.join(CST.PATH_DATA, f"{CST.HISTORY}.txt")
        self.path_vectordb = os.path.join(CST.PATH_DATA, CST.VECTORDB)
        self.path_title_image = os.path.join(CST.PATH_IMAGES, f"{CST.TITLE_IMAGE}.png")

        self.embeddings = HuggingFaceEmbeddings(model_name=CST.EMBEDDING_MODEL)
