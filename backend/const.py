class CST:
    """Constants"""

    # Assistant Name
    NAME = "Alma"

    # base url
    BASE_URL = "https://www.thebrokebackpacker.com"

    # paths
    PATH_DATA = "backend/data"
    PATH_IMAGES = "backend/images"
    PATH_UI = "backend/ui"

    # file names
    COUNTRIES = "countries"
    URLS = "guide_urls"
    HISTORY = "history"
    VECTORDB = "db"
    TITLE_IMAGE = "title_image"
    ASSISTANT_ICON = "assistant_icon"
    SIDEBAR = "sidebar"

    # user agent
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

    # request timeout for loading html content
    REQUEST_TIMEOUT = 15

    # Document splitter chunk size
    CHUNK_SIZE = 500

    # Document splitter chunk overlap
    CHUNK_OVERLAP = 0

    # Max number of documents to return from retriever
    K = 4

    # Retrieval Similarity Threshold
    THRESHOLD = 1.3

    # Embedding model for chunks
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM
    LLM = "mistralai/Mistral-Nemo-Instruct-2407"
    # LLM = "QuantFactory/Mistral-Nemo-Instruct-2407-GGUF"

    # Max new tokens for LLM to return
    MAX_NEW_TOKENS = 1024

    # LLM Temperature
    TEMPERATURE = 0.1

    QA_SYS_PROMPT = (
        "Your name is Alma. "
        "You're a female travel assistant with extensive backpacking experience, who can provide tips, help, advice and recommendations for other backpackers."
        "\n\n"
        "Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. "
        "Do not hesitate to paraphrase the retrieved context WORD FOR WORD to formulate your answer. in fact it is ENCOURAGED. "
        "Always elaborate. "
        "Do NOT answer questions non-related to travel or yourself, politely refuse to answer. "
        "Use colloquial language. "
        "Do not sugar-coat anything. Tell things as they are. Do not be misleading. "
        "Do not hesitate to add a touch of wit and humour, within moderation. "
        "You do not have to be PG-friendly. Expletives and taboo topics are allowed."
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
