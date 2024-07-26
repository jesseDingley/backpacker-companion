from backend.vectorizer import Vectorizer

test_vectorizer = Vectorizer()


def test_normalize_country():
    """
    Tests normalize_country()
    """
    assert test_vectorizer.normalize_country("Oman") == "oman"
    assert test_vectorizer.normalize_country("St Kitts & Nevis") == "st-kitts-and-nevis"
    assert test_vectorizer.normalize_country("Saudi Arabia") == "saudi-arabia"


def test_country_url_exists():
    """
    Tests country_url_exists()
    """
    urls = [
        "https://website.com/is-bangladesh-worth-visiting",
        "https://website.com/backpacking-cambodia",
        "https://website.com/backpacking-venezuela-travel-guide",
        "https://website.com/vietnam-backpacking",
    ]
    assert test_vectorizer.country_url_exists("venezuela", urls)
    assert test_vectorizer.country_url_exists("cambodia", urls)
    assert test_vectorizer.country_url_exists("bangladesh", urls)
    assert not test_vectorizer.country_url_exists("united-kingdom", urls)
    assert not test_vectorizer.country_url_exists("canada", [])
    assert test_vectorizer.country_url_exists("vietnam", urls)
