# Backpacker Companion

Travel & Backpacking companion RAG Chatbot deployed as a Streamlit web application.

[Web App link](https://backpacker-companion.streamlit.app)

## About

The RAG system leverages chunks of blogs scraped from The Broke Backpacker, which are saved to a Chroma collection deployed to GCP as a Cloud Run service.

I implement a hybrid retriever (BM25 + Vector Search) that is deployed as a FastAPI API to another Cloud Run service. The BM25 component of the retriever implemented with Llamaindex leverages the plain text blog chunks. The Vector Search component of the retriever calls the Chroma service.

User authentication is managed by Sign-in with Google and users will be forcefully logged out after one hour (Google ID Token expiry time).

LangChain is used for LLM logic and prompt engineering; ChromaDB for vector store and search; Llamaindex for BM25 KW search; Hugging Face Inference Endpoints (prod) / Ollama (local) for the LLM endpoints; and Streamlit for a simple front-end.

**Notes**:  

- *The code in this branch (main) is used in production.*
- *["local"](https://github.com/jesseDingley/backpacker-companion/tree/local) contains a simple system capable of running locally.*
- *["hybrid-retriever-no-jwt"](https://github.com/jesseDingley/backpacker-companion/tree/hybrid-retriever-no-jwt) contains hybrid retriever code.*
- *LLM runs on Ollama when executed locally.*


## Developer Notes

### Install & Run the Streamlit web app

```
$ make rerun
```

### Build the Hybrid Retriever

1. Deploy Chroma as a Google Cloud Run service.

```
$ make deploy-chroma \ 
    SERVICE_NAME=chroma \
    SERVICE_ACCOUNT_NAME=<gcp-service-account-name> \
    BUCKET_NAME=<storage-bucket-name> \
    PROJECT_ID=<project-id>
```
This also creates and mounts a GCS bucket directly to the Cloud Run container that will later be used to store the collection. Check out [this repo](https://github.com/HerveMignot/chromadb-on-gcp) for more info.

2. Create and upload data to a Chroma collection.

```
$ make create-collection
```

This command will overwrite any existing collection in the previously created Chroma service with a new one. The process involves scraping, chunking and vectorizing all blog posts from The Broke Backpacker and uploading them to the collection. The data will be saved to the bucket.

3. Deploy a fast Chroma service.

```
$ make deploy-chroma-copy \ 
    SERVICE_NAME=chroma \
    SERVICE_ACCOUNT_NAME=<gcp-service-account-name> \
    BUCKET_NAME=<storage-bucket-name> \
    PROJECT_ID=<project-id>
```

This deploys a second Chroma service that on boot will copy and download the embeddings and Chroma index files from the bucket to the fast service's container.

**NOTE**: To scrape the blogs again and update the collection, simply rerun  `make create-collection` and then reboot the fast Chroma service.

4. Switch to the `hybrid-retriever-no-jwt` branch .

5. Create Llamaindex Docstore for BM25 retriever

```
$ make create-docstore
```

6. Deploy the Hybrid Retriever as a Google Cloud Run service.

```
$ cd backend
$ gcloud run deploy --source . --memory 4Gi
```

When prompted give the service the name "hretriever".

**NOTE**: This always has to be done after creating the Docstore even if the service already exists.

