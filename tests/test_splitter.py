from backend.components.splitter import GuideTextSplitter

test_guide_url_splitter = GuideTextSplitter()


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

sample_page_content_w_out_tags = """This is heading 2:

Some text ...

This is heading 2:

Some text ...

This is heading 3:

This is heading 3 again:

Some text ...

This is heading 3 again again"""


def test_remove_tags():
    """
    Tests remove_tags()
    """
    assert (
        test_guide_url_splitter.remove_tags(sample_page_content_w_tags)
        == sample_page_content_w_out_tags
    )
