class CST:
    """Constants"""

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

    # Max new tokens for LLM to return
    MAX_NEW_TOKENS = 1024
    
    # LLM Temperature
    TEMPERATURE = 0.1

    # Max number of turns to store in memory
    # A turn is a (user, assistant) pair.
    MAX_TURNS = 10