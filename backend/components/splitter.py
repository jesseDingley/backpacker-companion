from typing import Any, Iterable, List
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class GuideTextSplitter(RecursiveCharacterTextSplitter):

    def __init__(
        self,
        separators: List[str] | None = [
            r"<h\d>.+</h\d>\n\n",
        ],
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any
    ) -> None:

        super().__init__(separators, keep_separator, is_separator_regex, **kwargs)

    @staticmethod
    def remove_tags(page_content: str) -> str:
        """
        Removes header tags from plain text page content

        Example: "<h3> Title </h3> bla bla" => "Title  bla bla"

        Args:
            page_content (str): plain text page content

        Returns:
            str: page content without tags
        """
        return re.sub(r"<h\d>(.+)</h\d>", r"\1:", page_content).strip()

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """
        Custom split_documents method that removes headers tags.
        """

        documents = super().split_documents(documents)

        for doc in documents:
            doc.page_content = GuideTextSplitter.remove_tags(doc.page_content)

        return documents
