from typing import Iterable, List, Dict, Tuple
import re
from langchain_core.documents import Document
from uuid import uuid4

class PostTextSplitter:
    """Class for splitting Loaded posts into chunks."""

    def __init__(
        self,
        separator: str = r"(?=<h)",
    ) -> None:

        self.separator = separator
        self.child_separator = "[SEP]"

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
    
    @staticmethod
    def remove_promos(page_content: str) -> str:
        """
        Removes Promos from chunk.

        Args:
            page_content (str): plain text chunk page content (no tags)

        Returns:
            str: chunk without promo
        """
        promos = [
            "\n\nUnlock Our GREATEST Travel Secrets",
            "\n\nOur GREATEST Travel Secrets",
            "\n\nDo You Want to Travel FOREVER",
            "\n\nWant to save money on accommodation?",
            "\n\nClick the button below",
            "\n\n*Author's Note:",
        ]
        for promo in promos:
            if promo in page_content:
                return page_content.split(promo)[0]
        return page_content
    
    @staticmethod
    def remove_commission_paragraph(page_content: str) -> str:
        """
        Removes commission paragraph from chunk.

        Args:
            page_content (str): plain text chunk page content (no tags)

        Returns:
            str: chunk without com par
        """
        commission_paragraph = "\n\nThe Broke Backpacker is supported by you. Clicking through our links may earn us a small affiliate commission, and that's what allows us to keep producing free content  Learn more."
        return page_content.replace(commission_paragraph, "")
    

    @staticmethod
    def remove_duplicate_sections(page_content: str) -> str:
        """
        Removes duplicated paragraphs that are separated by a double newline.

        Example:
            "
            Lowland Laos (Feb - April) : The temperatures are starting to climb.
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            : The temperatures are starting to climb. Highlands of Laos (November - January) : Pretty chill.
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            "
            =>
            "
            Lowland Laos (Feb - April) : The temperatures are starting to climb.
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            Highlands of Laos (November - January) : Pretty chill.
            "

        Args:
            page_content (str): plain text page content
        
        Returns:
            str: The cleaned-up text with duplicates removed.
        """
        # Regular expression to detect repeated paragraphs
        pattern = re.compile(r"(.*?)(\n\n\1)+", re.DOTALL)
        
        # Keep replacing matches until no more duplicates are found
        while True:
            new_page_content = re.sub(pattern, r"\1\n\n", page_content)
            new_page_content = re.sub(r"(\n\n)\s", r"\1", new_page_content)
            if new_page_content == page_content:
                break  # Stop if no changes were made
            page_content = new_page_content
        
        return page_content.strip()
    
    @staticmethod
    def is_buy_us_a_coffee_chunk(chunk: Document) -> bool:
        """Returns True if chunk is a 'buy us a coffee' chunk."""
        return chunk.page_content.strip().lower().startswith("buy us a coffee")
    
    @staticmethod
    def clean_chunk_titles(page_content: str) -> str:
        """
        Cleans chunk titles (
            replace :: > :
            replace ?: > ?
            replace !: > !
        )
        """
        return re.sub(r"([:\?\!]):", r"\1", page_content)
    
    @staticmethod
    def remove_fluff(page_content: str) -> str:
        """Removes fluff"""
        page_content = page_content.replace("CHECK TOP HOSTEL", "")
        page_content = page_content.replace("CHECK TOP HOTEL", "")
        page_content = page_content.replace("CHECK TOP AIRBNB", "")
        return page_content
    
    @staticmethod 
    def clean(page_content: str) -> str:
        """Cleans chunk page content."""
        page_content = PostTextSplitter.remove_tags(page_content)
        page_content = PostTextSplitter.remove_promos(page_content)
        page_content = PostTextSplitter.remove_commission_paragraph(page_content)
        page_content = PostTextSplitter.remove_duplicate_sections(page_content)
        page_content = PostTextSplitter.clean_chunk_titles(page_content)
        page_content = PostTextSplitter.remove_fluff(page_content)
        return page_content

    @staticmethod
    def add_title(page_content: str, title: str) -> str:
        """
        Adds title of article to Document / Chunk page_content, for better retrieval.

        Args:
            page_content (str): plain text page content
            title (str): source article title

        Returns:
            str: modded page_content
        """
        title = re.sub(r"\([^()]*\)", "", title).strip()
        return page_content + f" (taken from '{title}')"
    
    def split_documents(self, documents: Iterable[Document]) -> Tuple[List[Document], Dict[str, int]]:
        """
        Splits processed documents/blogs into chunks, 
        and creates a dict that maps chunk IDs to Document [h2 -> h3] mappings IDs.

        Example:

        chunk2mapping = {
            "id1": 0, # chunk 1 is associated with document 0
            "id2": 0, # chunk 2 is associated with document 0
            "id3": 1, # chunk 3 is associated with document 1
            "id4": 1, # chunk 4 is associated with document 1
            "id5": 1, # chunk 5 is associated with document 1
            ...
        }

        Args:
            documents: Iterable[Document]: docs to split

        Returns:
            Tuple[List[Document], Dict[str, int]]]: chunks and mapping
        """
        chunks = []
        chunk2mapping = {}
        for i, current_doc in enumerate(documents):
            splitted_current_doc_page_content = re.split(self.separator, current_doc.page_content)
            new_chunks = []
            for page_content in splitted_current_doc_page_content: 
                new_chunk_id = str(uuid4())
                new_chunk = Document(
                    page_content=page_content, 
                    metadata=current_doc.metadata.copy(), 
                )
                new_chunk.metadata["id"] = new_chunk_id # because similarity_search_with_score() from langchain_chroma doesn't return doc ids :(
                chunk2mapping[new_chunk_id] = i
                new_chunks.append(new_chunk)
            chunks += new_chunks
        return chunks, chunk2mapping

    def split_and_process_documents(self, documents: Iterable[Document], mappings: List[Dict[str, List[str]]]) -> List[Document]:
        """
        Splits and processes documents.

        Args:
            documents: Iterable[Document]: list of documents to split and process
            mappings: List[Dict[str, List[str]]]: list of h2->h3 mappings for each document

        Returns:
            List[Document]: chunks
        """

        chunks, chunk2mapping = self.split_documents(documents)

        for chunk in chunks:

            # clean
            chunk.page_content = PostTextSplitter.clean(chunk.page_content)
            del chunk.metadata["authors"]
            del chunk.metadata["publish_date"]
    
        # update metadata
        for chunk in chunks:
            chunk.metadata["children"] = ""
            for h2_title, h3_subtitles in mappings[chunk2mapping[chunk.metadata["id"]]].items():
                is_h3_chunk = any(
                    h3_subtitle in chunk.page_content for h3_subtitle in h3_subtitles
                )
                if h2_title in chunk.page_content and not is_h3_chunk:
                    for h3_subtitle in h3_subtitles:
                        for sub_doc in chunks:
                            if h2_title + " - " + h3_subtitle in sub_doc.page_content:
                                chunk.metadata["children"] += f"{sub_doc.metadata['id']}{self.child_separator}"

            chunk.metadata["children"] = re.sub(
                r"{}$".format(re.escape(self.child_separator)), 
                r"" ,
                chunk.metadata["children"]
            )

        # filter
        chunks = [
            chunk for chunk in chunks if (
                not (
                    PostTextSplitter.is_buy_us_a_coffee_chunk(chunk)
                ) and 
                not (
                    len(chunk.page_content.split(" ")) < 20 and 
                    chunk.metadata["children"] == ""
                )
            )
        ]

        return chunks
