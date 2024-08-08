class CST:
    """Constants"""

    # Assistant Name
    NAME = "Alma"

    # base url
    BASE_URL = "https://www.thebrokebackpacker.com"

    # paths
    PATH_DATA = "backend/data"
    PATH_IMAGES = "backend/images"

    # file names
    COUNTRIES = "countries"
    URLS = "guide_urls"
    HISTORY = "history"
    VECTORDB = "db"
    TITLE_IMAGE = "title_image"

    # user agent
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

    # request timeout for loading html content
    REQUEST_TIMEOUT = 15

    # Document splitter chunk size
    CHUNK_SIZE = 500

    # Document splitter chunk overlap
    CHUNK_OVERLAP = 0

    # Embedding model for chunks
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM
    LLM = "mistralai/Mistral-Nemo-Instruct-2407"
    # LLM = "QuantFactory/Mistral-Nemo-Instruct-2407-GGUF"

    QA_SYS_PROMPT = (
        "Your name is Alma, you're an experienced backpacker who can provide tips, help and recommendations for other backpackers."
        "\n\n"
        "Answer any use questions based solely on the context below. "
        "If you are not capable of providing a thorough and well-sourced answer, do not try to. "
        "If any use questions are out of the context of travel and backpacking, politely decline to continue conversation. "
        "Do not hesitate to add a touch of wit and humour, within moderation."
        "\n\n"
        "<context>"
        "\n"
        "{context}"
        "\n"
        "</context>"
    )

    REPHRASE_SYS_PROMPT = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # UI CSS
    MESSAGE_ALIGNMENT = "flex-end"
    MESSAGE_BG_COLOR = "linear-gradient(135deg, #00B2FF 0%, #006AFF 100%)"
