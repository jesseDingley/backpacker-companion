from backend.vectorizer import Vectorizer

def init_vectordb() -> None:
    """Init vector db from 0."""
    Vectorizer().run_from_zero()

def update_vectordb() -> None:
    """Update vector db."""
    Vectorizer().run_update()