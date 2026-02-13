def init_vectordb() -> None:
    """Init vector db from 0."""
    from backend.components.vectorizer import Vectorizer
    Vectorizer().run_from_zero()


def update_vectordb() -> None:
    """Update vector db."""
    from backend.components.vectorizer import Vectorizer
    Vectorizer().run_update()


def run() -> None:
    """Runs streamlit app."""
    from backend.chat import Chat
    Chat().run_app_safe()
