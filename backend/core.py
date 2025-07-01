from backend.components.vectorizer import Vectorizer
from backend.utils.utils import DocStoreManager
from backend.chat import Chat


def init_vectordb() -> None:
    """Init vector db from 0."""
    Vectorizer().run_from_zero()


def update_vectordb() -> None:
    """Update vector db."""
    Vectorizer().run_update()

def init_docstore() -> None:
    """Init docstore."""
    DocStoreManager().create_docstore_end_to_end()

def run() -> None:
    """Runs streamlit app."""
    Chat().run_app()
