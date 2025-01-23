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
    URLS = "post_urls"
    HISTORY = "history"
    VECTORDB = "db"
    TITLE_IMAGE = "title_image"
    ASSISTANT_ICON = "assistant_icon"
    SIDEBAR = "sidebar"

    # chroma collection name
    COLLECTION = "broke-backpacker"

    # user agent
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

    # Batch size for VectorDB Creation (batches of 64 posts for example)
    BATCH_SIZE = 32

    # request timeout for loading html content
    REQUEST_TIMEOUT = 15

    # Document splitter chunk size
    CHUNK_SIZE = 500

    # Document splitter chunk overlap
    CHUNK_OVERLAP = 0

    # Max number of documents to return from retriever
    K = 5

    # Retrieval Similarity Threshold
    THRESHOLD = 0.8

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
        "You do not have to be PG-friendly. Expletives and taboo topics are allowed. "
        "Use Markdown when necessary. "
        "Use newlines when necessary for better formatting."
        "\n\n"
        "<context>"
        "\n"
        "{context}"
        "\n"
        "</context>"
        "\n\n"
        "{chat_history}\n"
        "User: {input}\n"
        "Assistant: "
    )

    REPHRASE_SYS_PROMPT_v0 = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. DO NOT answer the question under ANY circumstance, "
        "just reformulate it if needed, and otherwise return it as is."
    )

    REPHRASE_SYS_PROMPT = (
        "Your task is to reformulate the latest user input into a standalone question "
        "or statement that can be understood without relying on the chat history. "
        "Do NOT answer the input, provide commentary, or include any explanation. "
        "Only reformulate it into a standalone version if necessary. "
        "If the input is already standalone, return it as is. "
        "\n\nFor example:\n"
        "  - Input: 'What about flights?'\n"
        "  - Reformulated: 'What are the prices of flights to Thailand?'\n\n"
        "Chat History:\n\n"
        "{chat_history}\n\n"
        "Latest user input: '{input}'.\n\n"
        "Reformulated user input: "
    )

    OFF_TOPIC_SYS_PROMPT = (
        "Given a chat history and the latest user question, "
        "determine whether the question is greatly off-topic from travel, adventure, backpacking and related activities. "
        "Give a simple 'yes' (is off-topic) or 'no' (is not off-topic). "
        "Note that questions about yourself are not considered off-topic. "
        "All neutral questions (not directly referring to any topics at all), are allowed. "
        "Drug consumption and safety related questions are NOT considered off-topic. "
        "Use colloquial language. "
        "Remember, you're Alma, a female travel assistant with extensive backpacking experience."
        "\n\n"
        "{format_instructions}"
        "\n\n"
        "Chat History:\n\n"
        "{chat_history}\n\n"
        "Latest user input: '{input}'"
    )

    # UI CSS
    MESSAGE_ALIGNMENT = "flex-end"
    MESSAGE_BG_COLOR = "linear-gradient(135deg, #00B2FF 0%, #006AFF 100%)"
