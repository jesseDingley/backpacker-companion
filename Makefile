install:
	pip install -e .

run:
	streamlit run app.py

rerun:
	pip install -e .
	streamlit run app.py

launch-api:
	python backend/retriever_api/retriever_api.py

deploy-chroma:
	export GOOGLE_APPLICATION_CREDENTIALS="gcloud-credentials.json"
	terraform plan -var-file chroma.tfvars
	terraform apply -var-file chroma.tfvars

destroy-chroma:
	export GOOGLE_APPLICATION_CREDENTIALS="gcloud-credentials.json"
	terraform destroy -var-file chroma.tfvars