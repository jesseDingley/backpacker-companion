from backend.guide_url_loader import GuideURLLoader

test_guide_url_loader = GuideURLLoader(urls=[])


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
