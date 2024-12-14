install:
	pip install -e .

run:
	streamlit run app.py

deploy-chroma:
	terraform plan -var-file chroma.tfvars
	terraform apply -var-file chroma.tfvars

destroy-chroma:
	terraform destroy -var-file chroma.tfvars