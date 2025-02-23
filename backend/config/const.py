class CST:
    """Constants"""

    # Batch size for VectorDB Creation (batches of 64 posts for example)
    BATCH_SIZE = 16

    # request timeout for loading html content
    REQUEST_TIMEOUT = 15

    # Max number of documents to return from retriever
    K = 7

    # Retrieval Similarity Threshold
    THRESHOLD = 0.7

    # Max new tokens for LLM to return
    MAX_NEW_TOKENS = 1024
    
    # LLM Temperature
    TEMPERATURE = 0.8

    # Max number of turns to store in memory
    # A turn is a (user, assistant) pair.
    MAX_TURNS = 10