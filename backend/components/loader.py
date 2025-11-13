from typing import Any, List, Dict, Tuple
import warnings
import re
from backend.components.custom_newsurlloader import CustomNewsURLLoader
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from unidecode import unidecode
import logging
from newspaper import Article


class PostURLLoader(CustomNewsURLLoader):
    """
    Custom Document loader for Post URLs.
    """

    @staticmethod
    def remove_image_captions(page_content: str, html_content: str) -> str:
        """
        Removes image captions from page content.

        Args:
            page_content (str): plain text page content
            html_content (str): raw html page content

        Returns:
            str: updated page content
        """
        soup = BeautifulSoup(html_content, features="lxml")
        img_caps = soup.find_all("figcaption", class_="wp-element-caption")
        for img_cap in img_caps:
            l = unidecode(" ".join([elt.strip() for elt in list(img_cap.stripped_strings)]))
            l = re.sub(r"\(\s*(\w+)\s*\)", r"(\1)", l)
            len_now = len(page_content)
            page_content = page_content.replace(l, "")
            if len(page_content) == len_now:
                l = re.sub(r"\s*(Image|Photo)", r"\n\n\1", l)
                page_content = page_content.replace(l, "")
        return page_content
    
    @staticmethod
    def remove_product_plugs(page_content: str, html_content: str) -> str:
        """
        Removes product plugs from page content

        Args:
            page_content (str): plain text page content
            html_content (str): raw html page content

        Returns:
            str: updated page content
        """
        soup = BeautifulSoup(html_content, features="lxml")
        plug_divs = soup.find_all("div", class_="product-plug")
        for plug in plug_divs:
            article = Article("")
            article.set_html(f"<h1></h1><p>{plug.get_text()}</p>")
            article.parse()
            normalized_plug_text = re.sub(
                r"(\w)([!?.]+)(\w)", 
                r"\1\2 \3", 
                unidecode(article.text)
            )
            page_content = page_content.replace(normalized_plug_text, "")
        return page_content
    
    @staticmethod
    def normalize_page_content(page_content: str, html_content: str) -> str:
        """
        Normalizes page content:
            > decodes it
            > trims it
            > removes image captions and product plugs
        
        Args:
            page_content (str): page content
            html_content (str): raw html

        Returns:
            str: normalized page content
        """
        norm_page_content = unidecode(page_content)
        norm_page_content = norm_page_content.split("\n\nMade it this far?")[0]
        norm_page_content = PostURLLoader.remove_image_captions(norm_page_content, html_content)
        norm_page_content = PostURLLoader.remove_product_plugs(norm_page_content, html_content)
        return norm_page_content
                
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

            headers_dict[header_type] = []

            for header in soup.find_all(header_type, class_="wp-block-heading"):

                if header.find("a") is None or header.find("a").text.strip() == "":
                    title_to_append = header.text
                else:
                    title_to_append = header.find("a").text

                headers_dict[header_type].append(
                    unidecode(title_to_append).strip()
                )

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
                re_expr = r"\n\n({})\n\n".format(re.escape(header))
                if len(re.findall(re_expr, unidecode(page_content))) > 1:
                    logging.warning(f"Header <<<{header}>>> appears more than once.")
                page_content = re.sub(
                    re_expr,
                    r"\n\n<{0}>{1}</{0}>\n\n".format(header_type, header),
                    unidecode(page_content),
                )
        return page_content
    
    @staticmethod
    def prepend_h2_title_to_h3_chunks(page_content: str) -> str:
        """
        Prepends the title of parent h2 sections to child h3 chunks.

        Example:

        "
        <h2>Best Places to Visit in Vietnam</h2>
        As I mentioned earlier, backpacking Vietnam is ...
        <h3>Backpacking Hanoi</h3>
        One of my favourite cities in all of ...
        <h3>Backpacking Sapa</h3>
        An explorers paradise, ...
        "

        =>

        "
        <h2>Best Places to Visit in Vietnam</h2>
        As I mentioned earlier, backpacking Vietnam is ...
        <h3>Best Places to Visit in Vietnam: Backpacking Hanoi</h3>
        One of my favourite cities in all of ...
        <h3>Best Places to Visit in Vietnam: Backpacking Sapa</h3>
        An explorers paradise, ...
        "

        Args:
            page_content (str): plain text page content with tags

        Returns:
            str: page content with prepended titles
        """

        h2_sections = page_content.split("<h2>")

        # if there are no h2 sections
        if len(h2_sections) == 1:
            return page_content
        
        # filter out blog intro
        if not "</h2>" in h2_sections[0]:
            h2_sections = h2_sections[1:]

        for h2_section in h2_sections:

            # check it is actually a h2 section
            assert h2_section.count("</h2>") == 1

            h3_sections = h2_section.split("<h3>")

            # if there are no subsections
            if len(h3_sections) == 1:
                continue

            # filter out subsection intro
            if not "</h3>" in h3_sections[0]:
                h3_sections = h3_sections[1:]

            # one subsection does not count
            #if len(h3_sections) == 1:
            #    continue

            # get parent title
            h2_title = h2_section.split("</h2>")[0].strip()
        
            for h3_section in h3_sections:

                # check it is actually a h3 section
                assert h3_section.count("</h3>") == 1

                if h3_section.split("</h3>")[0].strip() == "Final Thoughts" and len(h3_sections) == 1:
                    break

                # prepend
                new_h3_section = h2_title + " - " + h3_section
                page_content = page_content.replace(h3_section, new_h3_section)

        return page_content
    
    @staticmethod
    def get_h2_to_h3_mapping(page_content: str) -> Dict[str, List[str]]:
        """
        Returns dict where keys are h2 titles and values lists of h3 sub titles

        Args:
            page_content (str): plain text page content with tags and prepended h3 titles

        Returns:
            dict
        """

        mapping = {}

        h2_sections = page_content.split("<h2>")

        # if there are no h2 sections
        if len(h2_sections) == 1:
            return {}
        
        # filter out blog intro
        if not "</h2>" in h2_sections[0]:
            h2_sections = h2_sections[1:]

        for h2_section in h2_sections:

            # check it is actually a h2 section
            assert h2_section.count("</h2>") == 1

            # get parent title
            h2_title = h2_section.split("</h2>")[0].strip()
            mapping[h2_title] = []

            h3_sections = h2_section.split("<h3>")

            # if there are no subsections
            if len(h3_sections) == 1:
                continue

            # filter out subsection intro
            if not "</h3>" in h3_sections[0]:
                h3_sections = h3_sections[1:]

            # one subsection does not count
            #if len(h3_sections) == 1:
            #    d[h2_title] = False
            #    continue

            if len(h3_sections) == 1:
                assert h3_sections[0].count("</h3>") == 1
                if h3_sections[0].split("</h3>")[0].strip() == "Final Thoughts":
                    continue

            for h3_section in h3_sections:

                # check it is actually a h3 section
                assert h3_section.count("</h3>") == 1

                h3_title = h3_section.split("</h3>")[0]
                mapping[h2_title].append(h3_title)
            
            mapping[h2_title] = list(dict.fromkeys(mapping[h2_title]))

        return mapping

    def load(self) -> Tuple[List[Document], List[Dict[str, List[str]]]]:
        """
        Custom load method
        """

        mappings = []

        documents_and_htmls = super().load()

        documents = []
        htmls = []
        for doc, html in documents_and_htmls:
            documents.append(doc)
            htmls.append(html)

        for doc, html in zip(documents, htmls):

            if doc.page_content == "":

                warnings.warn(f"Document {doc.link} is empty!")

            else:

                # norm
                doc.page_content = PostURLLoader.normalize_page_content(doc.page_content, html)
                
                # tag page with headers
                doc_headers = PostURLLoader.get_headers_from_html(html)

                if doc_headers == {}:
                    logging.warning(f"NO HEADERS IN POST: {doc.metadata}")

                doc.page_content = PostURLLoader.tag_page_content(
                    page_content=doc.page_content, page_headers=doc_headers
                )

                mapping = PostURLLoader.get_h2_to_h3_mapping(doc.page_content)
                mappings.append(mapping)

                doc.page_content = PostURLLoader.prepend_h2_title_to_h3_chunks(
                    page_content=doc.page_content
                )

        return documents, mappings
