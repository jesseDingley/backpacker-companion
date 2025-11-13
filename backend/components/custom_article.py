from newspaper import Article
from bs4 import BeautifulSoup
import re

class CustomArticle(Article):
    """Custom Article that adds an extra cleaning step to html loading."""

    @staticmethod
    def clean_headers(soup: BeautifulSoup) -> None:
        """Cleans headers."""

        for header in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6"], 
            class_="wp-block-heading"
        ):
            
            header_text = header.get_text(strip=False).strip()
            new_header = soup.new_tag(header.name)
            new_header.string = header_text
            new_header["class"] = "wp-block-heading"
            header.replaceWith(new_header)

    @staticmethod
    def rm_quick_answers_section(soup: BeautifulSoup) -> None:
        """Removes 'quick answers' section. """

        for header in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6"], 
            class_="wp-block-heading"
        ):
            
            if re.search(
                r"^quick answer", 
                header.get_text(strip=False).strip(), 
                re.IGNORECASE
            ) is not None:

                current_h_type = int(header.name[-1])
                parent_h_types = ["h"+str(i) for i in range(current_h_type, 0, -1)]
                next_elt = header.find_next()
                elts_to_decomp = [next_elt]
                double_break = False

                while True:
                    if next_elt.name in parent_h_types:
                        break
                    for child in next_elt.findChildren():
                        if not child.name in parent_h_types:
                            double_break = True
                            break
                    if double_break:
                        break
                    next_elt = next_elt.find_next()
                    elts_to_decomp.append(next_elt)

                for elt in elts_to_decomp:
                    elt.decompose()

                header.decompose()

    @staticmethod
    def normalize_html(html: str) -> str:
        """
        Normalizes html for to make it easier for the NewsURLLoader().

        Args:
            html (str): raw html

        Returns:
            str: normalized html
        """

        soup = BeautifulSoup(html, features="lxml")

        CustomArticle.clean_headers(soup)
        CustomArticle.rm_quick_answers_section(soup)

        return str(soup)

    def download(self, input_html=None, title=None, recursion_counter=0):
        """Custom download method."""
        super().download(input_html, title, recursion_counter)
        new_html = CustomArticle.normalize_html(self.html)
        self.set_html(new_html)