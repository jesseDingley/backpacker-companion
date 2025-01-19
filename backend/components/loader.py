from typing import Any, List, Dict
import warnings
import re
from langchain_community.document_loaders import NewsURLLoader
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from unidecode import unidecode
import logging


class PostURLLoader(NewsURLLoader):
    """
    Custom Document loader for Post URLs.
    """

    def __init__(
        self,
        urls: List[str],
        text_mode: bool = True,
        nlp: bool = False,
        continue_on_failure: bool = True,
        show_progress_bar: bool = False,
        **newspaper_kwargs: Any,
    ) -> None:

        super().__init__(
            urls,
            text_mode,
            nlp,
            continue_on_failure,
            show_progress_bar,
            **newspaper_kwargs,
        )

        self.raw_html_loader = NewsURLLoader(urls=self.urls, text_mode=False)

    @staticmethod
    def clean_page_content(page_content: str) -> str:
        """
        Removes photo / image references and useless promo suffix from page content

        Args:
            page_content (str): page content

        Returns:
            str: cleaned page content
        """
        cleaned_page_content = re.sub(r"\n\n(Image|Photo):.+\n\n", "\n\n", page_content)
        return unidecode(cleaned_page_content.split("\n\nMade it this far?")[0])

    @staticmethod
    def get_headers_from_html(html_content: str) -> Dict[str, List[str]]:
        """
        Returns all headers from raw html content of web page,
        in a dict format where keys are header types and values are lists of header values

        Args:
            html_content (str): raw html content

        Returns:
           Dict[str, List[str]]
        """
        soup = BeautifulSoup(html_content, features="lxml")

        headers_dict = {}
        header_types = ["h2", "h3", "h4"]
        warning_count = 0
        for header_type in header_types:
            headers_dict[header_type] = [
                unidecode(header.text)
                for header in soup.find_all(header_type, class_="wp-block-heading")
            ]
            if headers_dict[header_type] == []:
                warning_count += 1

        if warning_count == len(header_types):
            return {}

        return headers_dict

    @staticmethod
    def tag_page_content(page_content: str, page_headers: Dict[str, List[str]]) -> str:
        """
        Tags headers in plain text page content.

        Example:   "\n\nThis is a header\n\n. Next paragraph ..."
                => "\n\n<h3>This is a header</h3>\n\n. Next paragraph ..."

        Args:
            page_content (str): page content
            page_headers (Dict[str, List[str]]): page headers extracted from raw html

        Returns:
            str: plain text page content w/ headers tagged
        """
        for header_type, headers in page_headers.items():
            for header in headers:
                page_content = re.sub(
                    r"\n\n({})\n\n".format(re.escape(header)),
                    r"\n\n<{0}>{1}</{0}>\n\n".format(header_type, header),
                    page_content,
                )
        return page_content

    def load(self) -> List[Document]:
        """
        Custom load method, removes photo / image references and useless promo suffix
        """
        documents = super().load()
        raw_documents = self.raw_html_loader.load()

        for doc, raw_doc in zip(documents, raw_documents):

            if doc.page_content == "":

                warnings.warn(f"Document {doc.link} is empty!")
            else:

                doc.page_content = PostURLLoader.clean_page_content(doc.page_content)
                
                doc_headers = PostURLLoader.get_headers_from_html(raw_doc.page_content)

                if doc_headers == {}:
                    logging.warning(f"NO HEADERS IN POST: {raw_doc.metadata}")

                doc.page_content = PostURLLoader.tag_page_content(
                    page_content=doc.page_content, page_headers=doc_headers
                )

        return documents
