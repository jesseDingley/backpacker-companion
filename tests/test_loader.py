from backend.loader import GuideURLLoader

test_guide_url_loader = GuideURLLoader(urls=[])

sample_html = """
<!DOCTYPE html>
<html>
    <body>
        <h1>This is heading 1</h1>
        <h2 class="wp-block-heading">This is heading 2</h2>
        <h3 class="wp-block-heading">This is heading 3</h3>
        <h3 class="wp-block-heading">This is heading 3 again</h3>
        <h3>This is heading 3 again again</h3>
    </body>
</html>
"""

headers = {
    "h2": ["This is heading 2"],
    "h3": ["This is heading 3", "This is heading 3 again"],
}

sample_page_content = """

This is heading 2

Some text ...

This is heading 2

Some text ...

This is heading 3

This is heading 3 again

Some text ...

This is heading 3 again again
"""

sample_page_content_w_tags = """

<h2>This is heading 2</h2>

Some text ...

<h2>This is heading 2</h2>

Some text ...

<h3>This is heading 3</h3>

<h3>This is heading 3 again</h3>

Some text ...

This is heading 3 again again
"""


def test_clean_page_content():
    """
    Tests clean_page_content()
    """
    test_str_1 = "Some text"
    assert test_guide_url_loader.clean_page_content(test_str_1) == test_str_1

    test_str_2 = "Some text with Image: without \n"
    assert test_guide_url_loader.clean_page_content(test_str_2) == test_str_2

    test_str_3 = "Some text where we have an image \n\nPhoto: CJ\n\nok son here's another: \n\nPhoto: CJ\n\n. What a pic"
    assert (
        test_guide_url_loader.clean_page_content(test_str_3)
        == "Some text where we have an image \n\nok son here's another: \n\n. What a pic"
    )

    test_str_4 = "Some text where we have an image \n\nPhoto: CJ\n\nok son here's another: \n\nPhoto: CJ\n\n. What a pic\n\nMade it this far? Well done."
    assert (
        test_guide_url_loader.clean_page_content(test_str_4)
        == "Some text where we have an image \n\nok son here's another: \n\n. What a pic"
    )


def test_get_headers_from_html():
    """
    Tests get_headers_from_html()
    """
    assert test_guide_url_loader.get_headers_from_html(sample_html) == headers


def test_tag_page_content():
    """
    Tests tag_page_content()
    """
    print(test_guide_url_loader.tag_page_content(sample_page_content, headers))
    assert (
        test_guide_url_loader.tag_page_content(sample_page_content, headers)
        == sample_page_content_w_tags
    )
