from typing import List
import requests
import os
from tqdm import tqdm
from backend.guide_url_loader import GuideURLLoader
from langchain_core.documents.base import Document


class Vectorizer:
    """
    Class to scrape, chunk and vectorize Broke Backpacker website.

    Attributes
    ------------

    base_url (str): brokebackpacker home page url
    path_data (str): relative path to folder containing data
    path_countries (str): relative path to txt of country names
    path_guide_urls (str): relative path to txt of country travel guide urls
    url_structures (List[str]): possible name combinations for a travel guide url
    headers (dict): https request header


    Methods
    ------------
    normalize_country(country: str) -> str: Normalizes country name.
    country_url_exists(country: str, urls: List[str]) -> bool: Determines if country is in one url from a list of urls.
    collect_urls(self) -> None: Collect all URLs to scrape, i.e. every backpacking guide by country, and saves them to disk.
    """

    def __init__(self) -> None:

        self.base_url = "https://www.thebrokebackpacker.com"

        self.path_data = "backend/data"
        self.path_countries = os.path.join(self.path_data, "countries.txt")
        self.path_guide_urls = os.path.join(self.path_data, "guide_urls.txt")

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

    def load_url_content(self) -> List[Document]:
        """
        Loads web page content from each guide url into a list of Langchain Documents

        Returns:
            List[Document]: list of Langchain Documents
        """
        with open(self.path_guide_urls, "r", encoding="utf8") as f:
            guide_urls = [url.replace("\n", "").strip() for url in f.readlines()]

        url_loader = GuideURLLoader(urls=guide_urls, show_progress_bar=True)

        return url_loader.load()

    def run(self):
        self.collect_urls()
        docs = self.load_url_content()
