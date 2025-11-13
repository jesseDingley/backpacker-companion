from langchain_community.document_loaders import NewsURLLoader
from typing import Iterator
from langchain_core.documents import Document
from typing import Tuple
import logging

class CustomNewsURLLoader(NewsURLLoader):
    """Custom NewsURLLoader to use CustomArticle."""

    def lazy_load(self) -> Iterator[Tuple[Document, str]]:
        """
        Overwritten lazy_load method that uses CustomArticle() instead of Article(),
        and returns html with the Document.
        """
        try:
            from backend.components.custom_article import CustomArticle
        except ImportError as e:
            raise ImportError(
                "Cannot import newspaper, please install with `pip install newspaper3k`"
            ) from e

        for url in self.urls:
            try:
                article = CustomArticle(url, **self.newspaper_kwargs)
                article.download()
                article.parse()

                if self.nlp:
                    article.nlp()

            except Exception as e:
                if self.continue_on_failure:
                    logging.error(f"Error fetching or processing {url}, exception: {e}")
                    continue
                else:
                    raise e

            metadata = {
                "title": getattr(article, "title", ""),
                "link": getattr(article, "url", getattr(article, "canonical_link", "")),
                "authors": getattr(article, "authors", []),
                "language": getattr(article, "meta_lang", ""),
                "description": getattr(article, "meta_description", ""),
                "publish_date": getattr(article, "publish_date", ""),
            }

            #if self.text_mode:
            #    content = article.text
            #else:
            #    content = article.html

            if self.nlp:
                metadata["keywords"] = getattr(article, "keywords", [])
                metadata["summary"] = getattr(article, "summary", "")

            yield (Document(page_content=article.text, metadata=metadata), article.html)
