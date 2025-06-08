# Backpacker Companion - Deployed Version

[Live Streamlit link](https://backpacker-companion.streamlit.app)


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

## Populating Chroma Collection

Run `$ make create-collection`

## How to run

If the repo is not currently deployed to streamlit and you wish to test it yourself you will need to:

1. Follow [this](https://docs.trychroma.com/deployment/gcp) guide to deploy Chroma to GCP, including the "Authentication with GCP" section. Afterwards, you should have `glcoud-credentials.json`, `chroma.tfvars`, `main.tf`, `terraform.tfstate` files at the root. Don't hesitate to refer back to the Makefile for deploy and destroy commands.

2. At the root of the repository, create a `.streamlit/secrets.toml` file like the following:

```
HUGGINGFACE_API_KEY = "<your-api-key>"
chroma_server_auth_credentials = "<your-token>"
chroma_ip = "<your-vm-ip-returned-by-terraform>"
```

3. `$ make install` to install the package

4. `$ init-vectordb` to create the Chroma DB, but you might want to limit the number of URLs to be processed, otherwise the process might take 3-4 hours.

5. `$ make run` to open the streamlit app.

## Room for improvement.

- Improve Sys Prompt.
- Rephrased input is computed twice.
- Test hybrid vector / kw search
- (Include full length blogs)
- Fix multiple header appearances (like in https://www.thebrokebackpacker.com/travel-insurance-mexico/)