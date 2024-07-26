from typing import List
import requests
import os
from tqdm import tqdm
from backend.loader import GuideURLLoader
from backend.splitter import GuideTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from newspaper import Config


class Vectorizer:
    """
    Class to scrape, chunk and vectorize Broke Backpacker website.

    Attributes
    ------------

    base_url (str): brokebackpacker home page url
    path_data (str): relative path to folder containing data
    path_countries (str): relative path to txt of country names
    path_guide_urls (str): relative path to txt of country travel guide urls
    path_vector_db (str): relative path to vector db
    url_structures (List[str]): possible name combinations for a travel guide url
    headers (dict): https request header
    newspaper_kwargs (dict): kwargs for document loader
    text_splitter (GuideTextSplitter): custom text splitter
    embeddings (HuggingFaceEmbeddings): embeddings model


    Methods
    ------------
    normalize_country(country: str) -> str: Normalizes country name.
    country_url_exists(country: str, urls: List[str]) -> bool: Determines if country is in one url from a list of urls.
    collect_urls(self) -> None: Collect all URLs to scrape, i.e. every backpacking guide by country, and saves them to disk.
    init_vector_db(self) -> None: Initializes Vector DB from content found from Guide URLs.
    """

    def __init__(
        self,
        request_timeout: int = 15,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """
        Init

        Args:
            request_timeout (int): request timeout for loading urls
            chunk_size (int): document splitting chunk size. Has little effect due to chosen separators.
            chunk_overlap (int): document splitting chunk overlap.
            embeddings_model (str): HF embedding model name
        """

        self.base_url = "https://www.thebrokebackpacker.com"

        self.path_data = "backend/data"
        self.path_countries = os.path.join(self.path_data, "countries.txt")
        self.path_guide_urls = os.path.join(self.path_data, "guide_urls.txt")
        self.path_vectordb = os.path.join(self.path_data, "db")

        self.url_structures = [
            "backpacking-{country}-travel-guide",
            "backpacking-{country}-budget-travel-guide",
            "backpacking-{country}-ultimate-travel-guide",
            "backpacking-{country}-ultimate-budget-travel-guide",
            "backpacking-{country}-destination-guide",
            "backpacking-{country}",
            "backpacking-{country}-on-a-budget",
            "backpacking-in-{country}",
            "{country}-backpacking",
            "is-{country}-worth-visiting",
        ]

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

        newspaper_config = Config()
        newspaper_config.browser_user_agent = self.headers["User-Agent"]
        newspaper_config.request_timeout = request_timeout

        self.newspaper_kwargs = {"config": newspaper_config}

        self.text_splitter = GuideTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

    @staticmethod
    def normalize_country(country: str) -> str:
        """
        Normalizes country name.

        Example: "San Marino" -> "san-marino"

        Args:
            country (str): raw country

        Returns:
            str: normalized country
        """
        return country.strip().lower().replace(" ", "-").replace("&", "and")

    @staticmethod
    def country_url_exists(country: str, urls: List[str]) -> bool:
        """
        Determines if country is in one url from a list of urls.

        Example: "canada" IN [
            "https://www.thebrokebackpacker.com/backpacking-canada-travel-guide",
            "https://www.thebrokebackpacker.com/backpacking-chile-travel-guide"
        ] -> TRUE

        Args:
            country (str): country
            urls (List[str]): list of urls

        Returns:
            bool: True if country is in one of the urls
        """
        if urls == []:
            return False
        return len([url for url in urls if f"-{country}" in url]) == 1

    def collect_urls(self, reset: bool = False) -> None:
        """
        Collect all URLs to scrape, i.e. every backpacking guide by country,
        and saves them to disk.

        If guide_urls.txt already exists, only new urls will be appended

        Args:
            reset (bool): if True, deletes current guide url txt file
        """

        if reset and os.path.exists(self.path_guide_urls):
            os.remove(self.path_guide_urls)

        valid_urls = []

        scraped_urls = []
        if os.path.exists(self.path_guide_urls):
            with open(self.path_guide_urls, "r", encoding="utf8") as f:
                scraped_urls = [url.replace("\n", "").strip() for url in f.readlines()]

        with open(self.path_countries, "r", encoding="utf8") as f:
            countries = f.readlines()

        normalized_countries = [
            Vectorizer.normalize_country(country) for country in countries
        ]

        for country in tqdm(normalized_countries):

            if Vectorizer.country_url_exists(country, scraped_urls):
                continue

            for struct in self.url_structures:

                url_to_try = os.path.join(self.base_url, struct.format(country=country))

                response = requests.get(url=url_to_try, headers=self.headers)

                if response.status_code == 200:
                    if "text/html" in response.headers.get("content-type"):
                        valid_urls.append(url_to_try)
                        break

        write_mode = "w" if scraped_urls == [] else "a"

        if valid_urls != []:
            with open(self.path_guide_urls, write_mode) as f:
                if write_mode == "a":
                    f.write("\n")
                f.write("\n".join(valid_urls))

        if valid_urls == [] and write_mode == "w":
            raise ValueError("No valid URLs found!")

    def init_vector_db(self) -> None:
        """
        Initializes Vector DB from content found from Guide URLs.
        """

        with open(self.path_guide_urls, "r", encoding="utf8") as f:
            guide_urls = [url.replace("\n", "").strip() for url in f.readlines()]

        url_loader = GuideURLLoader(
            urls=guide_urls, show_progress_bar=True, **self.newspaper_kwargs
        )

        docs = url_loader.load()

        texts = self.text_splitter.split_documents(docs)
        texts = filter_complex_metadata(texts)

        Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.path_vectordb,
        )
