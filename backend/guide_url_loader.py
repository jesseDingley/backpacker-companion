from typing import Any, List
import warnings
import re
from langchain_community.document_loaders import NewsURLLoader
from langchain_core.documents import Document


class GuideURLLoader(NewsURLLoader):
    """
    Custom Document loader for Guide URLs.
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
        return cleaned_page_content.split("\n\nMade it this far?")[0]

    def load(self) -> List[Document]:
        """
        Custom load method, removes photo / image references and useless promo suffix
        """
        documents = super().load()

        for i, doc in enumerate(documents):

            if doc.page_content == "":
                warnings.warn(f"Document {i} ({doc.link}) is empty!")
            else:
                doc.page_content = GuideURLLoader.clean_page_content(doc.page_content)

        return documents
