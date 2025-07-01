install:
	pip install -e .

create-collection:
	pip install -e .
	init-vectordb	

create-docstore:
	pip install -e .
	init-docstore

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