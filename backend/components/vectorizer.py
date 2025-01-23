from typing import List
import requests
import os
import logging
from time import time
from tqdm import tqdm
from backend.components.loader import PostURLLoader
from backend.components.splitter import PostTextSplitter
from backend.config.const import CST
from backend.base import Base
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from newspaper import Config
from bs4 import BeautifulSoup

class Vectorizer(Base):
    """
    Class to scrape, chunk and vectorize Broke Backpacker website 
    into a Chroma vector store uploaded to a gcloud vm.

    Attributes
    ------------
    headers (dict): https request header
    newspaper_kwargs (dict): kwargs for document loader
    text_splitter (PostTextSplitter): custom text splitter


    Methods
    ------------
    get_and_save_post_urls(self) -> None: Gets all post URLs from thebrokebackpacker sitemap, and saves them to backend/data/post_urls.txt
    process_and_upload_batch(self, batch: List[str], i: int) -> None: Processes and uploads batch of post urls to vector db. 
    init_vectordb(self) -> None: Initializes Vector DB.
    run(self, from_zero: bool) -> None: Runs vectorizer pipeline.
    run_from_zero(self) -> None: Runs the pipeline from zero.
    run_update(self) -> None: Runs the pipeline to update the existing vector db.
    """

    def __init__(self) -> None:
        super().__init__()

        self.headers = {"User-Agent": self.collection_config["USER_AGENT"]}

        newspaper_config = Config()
        newspaper_config.browser_user_agent = self.headers["User-Agent"]
        newspaper_config.request_timeout = CST.REQUEST_TIMEOUT
        self.newspaper_kwargs = {"config": newspaper_config}

        self.text_splitter = PostTextSplitter(
            chunk_size=CST.CHUNK_SIZE,
            chunk_overlap=CST.CHUNK_OVERLAP,
            length_function=len,
        )

    def get_and_save_post_urls(self) -> None:
        """
        Gets all post URLs from thebrokebackpacker sitemap,
        and saves them to backend/data/post_urls.txt

        Returns:
            List[str]: list of post urls.
        """

        logging.info("Collecting Post URLs...")

        urls = []

        sitemap_url = os.path.join(
            self.collection_config["BASE_URL"], 
            "sitemap"
        )

        response = requests.get(url=sitemap_url, headers=self.headers)

        if response.status_code == 200:

            soup = BeautifulSoup(response.text, 'html.parser')

            posts_ul = soup.find_all('ul', class_='sitemap-post')[0].find_all("li")

            for post_li in posts_ul:

                post_url = self.collection_config["BASE_URL"] + post_li.find("a").attrs["href"]

                if not post_url in [
                    "https://www.thebrokebackpacker.com/best-bivvy-bags/",
                    "https://www.thebrokebackpacker.com/privacy-policy/",
                    "https://www.thebrokebackpacker.com/barefoot-embrace-of-life-on-koh-lanta/",
                    "https://www.thebrokebackpacker.com/freely-travel-insurance-review/",
                ]:

                    urls.append(post_url)

            with open(self.paths["POST_URLS"], "w") as f:
                f.write("\n".join(urls))

            logging.info("Done.")

        else:

            raise Exception(f"Failed to load sitemap: {response.status_code}")
    
    def process_and_upload_batch(self, batch: List[str], i: int) -> None:
        """
        Processes and uploads batch of post urls to vector db. 
        
        i.e:
            > Loads URL content
            > Splits content into chunks
            > Clean chunks
            > Adds chunks to VectorDB

        Args:
            batch (List[str]): list of urls
            i (int): batch index
        """

        url_loader = PostURLLoader(
            urls=batch, show_progress_bar=True, **self.newspaper_kwargs
        )

        docs = url_loader.load()

        texts = self.text_splitter.split_documents(docs)
        texts = filter_complex_metadata(texts)

        if i == 0:
            
            # DELETE EXISTING COLLECTION

            try:
                self.chroma_client.delete_collection(
                    name=self.collection_config["NAME"]
                )
                logging.info(f"Deleted existing collection {self.collection_config['NAME']}.")
            except:
                pass

        # CREATE / ADD TO COLLECTION

        collection = Chroma(
            collection_name=self.collection_config["NAME"],
            client=self.chroma_client,
            embedding_function=self.embeddings,
            create_collection_if_not_exists=True
        )

        if i == 0:
            logging.info(f"Created collection {self.collection_config['NAME']}.")

        collection.add_documents(
            documents=texts
        )

    def init_vectordb(self) -> None:
        """
        Initializes Vector DB.

        More specifically, uploads documents in 
        """
        with open(self.paths["POST_URLS"], "r", encoding="utf8") as f:
            post_urls = [
                url.replace("\n", "").strip() for url in f.readlines()
            ]

        batches = [
            post_urls[i:i + CST.BATCH_SIZE] for i in range(
                0, 
                len(post_urls), 
                CST.BATCH_SIZE
            )
        ]

        num_batches = len(batches)

        logging.info("VectorDB creation initialized.")

        for i, batch in enumerate(tqdm(batches)):
            t0 = time()
            self.process_and_upload_batch(batch, i)
            t1 = time()
            delta_t = t1 - t0
            minutes, seconds = divmod(delta_t, 60)
            logging.info(f"Processed and Uploaded Batch {i+1}/{num_batches} in {int(minutes)} min {int(seconds)} sec.")

    def run(self, from_zero: bool) -> None:
        """
        Runs vectorizer pipeline.

        Setting from_zero to True will overwrite collected URLs and overwrite existing Vector DB
        Setting from_zero to False will append new URLs and update existing Vector DB

        Args:
            from_zero (bool): True to run from 0. False, to update
        """
        if from_zero:

            logging.info("Creating VectorDB from scratch.")
            self.get_and_save_post_urls()
            self.init_vectordb()
            logging.info("All batches uploaded and VectorDB successfully created.")

        else:
            logging.error("Cannot currently dynamically update VectorDB")

    def run_from_zero(self) -> None:
        """Runs the pipeline from zero."""
        self.run(from_zero=True)

    def run_update(self) -> None:
        """Runs the pipeline to update the existing vector db."""
        self.run(from_zero=False)
