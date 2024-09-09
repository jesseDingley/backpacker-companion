from backend.components.vectorizer import Vectorizer
from backend.chat import Chat


def init_vectordb() -> None:
    """Init vector db from 0."""
    Vectorizer().run_from_zero()


def update_vectordb() -> None:
    """Update vector db."""
    Vectorizer().run_update()


def run() -> None:
    """Runs streamlit app."""
    Chat().run_app()
