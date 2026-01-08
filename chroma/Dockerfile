FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir chromadb google-cloud-storage

COPY boot.py .

RUN mkdir /data

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["python", "boot.py"]