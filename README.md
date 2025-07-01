# Backpacker Companion - Hybrid Retriever Service

This branch contains the code for deploying the Hybrid Retriever to a Google Cloud Run Service.

## Deploying the service

To (re) deploy the Hybrid Retriever as a Google Cloud Run service, run

```
$ cd backend
$ gcloud run deploy --source . --memory 4Gi
```