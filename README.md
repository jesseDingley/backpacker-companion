# Backpacker Companion

Travel & Backpacking companion RAG Chatbot deployed as a Streamlit web application.

[Web App link](https://backpacker-companion.streamlit.app)

*Note:* may not be live.

## About

The RAG system leverages chunks of blogs scraped from [The Broke Backpacker](http://thebrokebackpacker.com/), which are saved to a Chroma collection deployed to GCP as a Cloud Run service.

Hybrid retrieval (BM25 + Vector Search) is implemented, for which a custom retriever is deployed to another Cloud Run service. The BM25 component of the retriever leverages the plain text blog chunks, and the Vector Search component is the Chroma service.

User authentication is managed by Sign-in with Google and users will be forcefully logged out after one hour (Google ID Token expiry time).

**Important Notes**:  

- *The code in this branch is used in production*
- *You cannot run this app locally unless you also:*
    - Have a HF Inference Endpoint / Ollama model downloaded
    - Deploy Chroma as a Cloud Run service 
    - Create & Populate a collection to the Chroma service.
    - Deploy a (hybrid) retriever to a Cloud Run service.

## Install & Run 

```
$ make rerun
```


## Deploying Chroma

To (re) deploy Chroma as a Google Cloud Run service, run

```
$ make deploy-chroma \ 
    SERVICE_NAME=chroma \
    SERVICE_ACCOUNT_NAME=<gcp-service-account-name> \
    BUCKET_NAME=<storage-bucket-name> \
    PROJECT_ID=<project-id>
```
(Mounts the GCS bucket directly to the Cloud Run container using. Slow. Inspired from https://github.com/HerveMignot/chromadb-on-gcp)

or 

```
$ make deploy-chroma-copy \ 
    SERVICE_NAME=chroma \
    SERVICE_ACCOUNT_NAME=<gcp-service-account-name> \
    BUCKET_NAME=<storage-bucket-name> \
    PROJECT_ID=<project-id>
```

(Downloads the entire Chroma collection from the GCS bucket to the container's local filesystem. This method is only to be used if the collection already exists. Attempting to write to the collection with this service will not update the collection in the GCS bucket, only the ephemeral copied collection to the service container.)

## Populating Chroma Collection

Run

```
$ make create-collection
```

to scrape, chunk, vectorize and add blogs to previously created Chroma service.

## Deploying Hybrid Retriever

See `hybrid-retriever-no-jwt` branch.

## Branches

- ["local"](https://github.com/jesseDingley/backpacker-companion/tree/local) contains a simple system capable of running locally.
- ["hybrid-retriever-no-jwt"](https://github.com/jesseDingley/backpacker-companion/tree/hybrid-retriever-no-jwt) contains hybrid retriever code.
