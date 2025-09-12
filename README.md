# Backpacker Companion - Deployed Version

[Streamlit link](https://backpacker-companion.streamlit.app)

## About

Travel & Backpacking Companion RAG Chat-Bot deployed as a Streamlit application.

The RAG system leverages chunks of blogs scraped from http://thebrokebackpacker.com/, which are saved to a Chroma collection deployed to a Google Cloud Run service.

A hybrid retriever is used (BM25 + Vector Search), which is deployed to another Google Cloud Run service. The BM25 component of the retriever leverages the plain text blog chunks, and the Vector Search component is the Chroma collection service.

User authentication is managed by Sign-in with Google and JWT tokens that are generated server-side to allow users to stay logged-in for longer, since Google ID tokens expire within one hour.

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
    SERVICE_NAME=<your-cloud-run-service-name> \
    SERVICE_ACCOUNT_NAME=<gcp-service-account-name> \
    BUCKET_NAME=<storage-bucket-name> \
    PROJECT_ID=<project-id>
```

See https://github.com/HerveMignot/chromadb-on-gcp for further details.

**NOTE:** This only deploys Chroma as a service, it does not create any collections nor add any documents.

## Populating Chroma Collection

Run

```
$ make create-collection
```

to scrape, chunk, vectorize and add blogs to previously created Chroma service.

## Deploying Hybrid Retriever

See `hybrid-retriever` branch.


## Room for improvement.

- Add step to check whether retrieval is necessary / make filter stricter