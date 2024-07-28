from typing import List
import requests
import os
import re
import shutil
from tqdm import tqdm
from datetime import datetime
from backend.loader import GuideURLLoader
from backend.splitter import GuideTextSplitter
from backend.const import CST
from backend.base import Base
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from newspaper import Config


class Vectorizer(Base):
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
    init_vector_db(self) -> None: Initializes Vector DB
    update_vector_db(self) -> None: Updates Vector DB with content from new URLs
    run(self, from_zero: bool) -> None: Runs vectorizer pipeline.
    run_from_zero(self) -> None: Runs the pipeline from zero.
    run_update(self) -> None: Runs the pipeline to update the existing vector db.
    """

    def __init__(self) -> None:
        super().__init__()

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

        self.headers = {"User-Agent": CST.USER_AGENT}

        newspaper_config = Config()
        newspaper_config.browser_user_agent = self.headers["User-Agent"]
        newspaper_config.request_timeout = CST.REQUEST_TIMEOUT
        self.newspaper_kwargs = {"config": newspaper_config}

        self.text_splitter = GuideTextSplitter(
            chunk_size=CST.CHUNK_SIZE,
            chunk_overlap=CST.CHUNK_OVERLAP,
            length_function=len,
        )

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
        return (
            len(
                [
                    url
                    for url in urls
                    if (f"-{country}" in url) or (f"{country}-" in url)
                ]
            )
            == 1
        )

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

        if reset and os.path.exists(self.path_history):
            os.remove(self.path_history)

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

                url_to_try = os.path.join(CST.BASE_URL, struct.format(country=country))

                response = requests.get(url=url_to_try, headers=self.headers)

                if response.status_code == 200:
                    if "text/html" in response.headers.get("content-type"):
                        valid_urls.append(url_to_try)
                        break

        write_mode = "w" if scraped_urls == [] else "a"

        if valid_urls != []:
            with open(self.path_guide_urls, write_mode) as f:
                if write_mode == "a":
                    now = datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d_%H:%M:%S")
                    f.write(f"\n{now_formatted}\n")
                f.write("\n".join(valid_urls))

        if valid_urls == [] and write_mode == "w":
            raise ValueError("No valid URLs found!")

    def init_vector_db(self) -> None:
        """
        Initializes Vector DB.
        """

        with open(self.path_guide_urls, "r", encoding="utf8") as f:
            guide_urls = [url.replace("\n", "").strip() for url in f.readlines()]

        url_loader = GuideURLLoader(
            urls=guide_urls, show_progress_bar=True, **self.newspaper_kwargs
        )

        docs = url_loader.load()

        texts = self.text_splitter.split_documents(docs)
        texts = filter_complex_metadata(texts)

        if os.path.exists(self.path_vectordb):
            shutil.rmtree(self.path_vectordb)

        Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.path_vectordb,
        )

    def update_vector_db(self) -> None:
        """
        Updates Vector DB with content from new URLs
        """

        # load all urls
        with open(self.path_guide_urls, "r", encoding="utf8") as f:
            guide_urls = [url.replace("\n", "").strip() for url in f.readlines()]

        # get dates of updates to url list
        split_points = [
            elt
            for elt in guide_urls
            if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}", elt) is not None
        ]

        # no new urls
        if split_points == []:
            return

        # load vectordb update history
        if os.path.exists(self.path_history):
            with open(self.path_history, "r", encoding="utf8") as f:
                history = [line.replace("\n", "").strip() for line in f.readlines()]

            # the new urls have already updated the vector db
            if history[-1] == split_points[-1]:
                return

        # get new urls
        split_index = [
            i for i, elt in enumerate(guide_urls) if elt == split_points[-1]
        ][0] + 1
        new_urls = guide_urls[split_index:]

        # create new docs

        url_loader = GuideURLLoader(
            urls=new_urls, show_progress_bar=True, **self.newspaper_kwargs
        )

        docs = url_loader.load()

        texts = self.text_splitter.split_documents(docs)
        texts = filter_complex_metadata(texts)

        # load vector db
        vectordb = Chroma(
            persist_directory=self.path_vectordb, embedding_function=self.embeddings
        )

        # update vector db
        vectordb.add_documents(texts)
        del vectordb

        # update history
        if not os.path.exists(self.path_history):
            with open(self.path_history, "w", encoding="utf8") as f:
                f.write(split_points[-1])
        else:
            with open(self.path_history, "a", encoding="utf8") as f:
                f.write(f"\n{split_points[-1]}")

    def run(self, from_zero: bool) -> None:
        """
        Runs vectorizer pipeline.

        Setting from_zero to True will overwrite collected URLs and overwrite existing Vector DB
        Setting from_zero to False will append new URLs and update existing Vector DB

        Args:
            from_zero (bool): True to run from 0, False, to update
        """
        if from_zero:
            self.collect_urls(reset=True)
            self.init_vector_db()
        else:
            self.collect_urls()
            self.update_vector_db()

    def run_from_zero(self) -> None:
        """Runs the pipeline from zero."""
        self.run(from_zero=True)

    def run_update(self) -> None:
        """Runs the pipeline to update the existing vector db."""
        self.run(from_zero=False)
