install:
	pip install -e .

create-collection:
	pip install -e .
	init-vectordb	

run:
	streamlit run app.py

rerun:
	pip install -e .
	streamlit run app.py

launch-api:
	python backend/retriever_api/retriever_api.py

deploy-chroma:
	SERVICE_NAME=$(SERVICE_NAME) SERVICE_ACCOUNT_NAME=$(SERVICE_ACCOUNT_NAME) BUCKET_NAME=$(BUCKET_NAME) PROJECT_ID=$(PROJECT_ID) \
	envsubst < chroma/deploy_chroma_template.yaml > chroma/deploy_chroma.yaml && \
	gcloud run services replace chroma/deploy_chroma.yaml  --project $(PROJECT_ID)

deploy-chroma-copy:
	gcloud builds submit --tag gcr.io/$(PROJECT_ID)/chroma-copy:latest chroma/
	SERVICE_NAME=$(SERVICE_NAME)-copy SERVICE_ACCOUNT_NAME=$(SERVICE_ACCOUNT_NAME) BUCKET_NAME=$(BUCKET_NAME) IMAGE_NAME=gcr.io/$(PROJECT_ID)/chroma-copy:latest \
	envsubst < chroma/deploy_chroma_copy_template.yaml > chroma/deploy_chroma_copy.yaml && \
	gcloud run services replace chroma/deploy_chroma_copy.yaml --project $(PROJECT_ID)