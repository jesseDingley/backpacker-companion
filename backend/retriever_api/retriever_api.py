from fastapi import FastAPI, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from hybrid_retriever import HybridRetriever
import uvicorn
import os
import signal
import streamlit as st

API_KEY = st.secrets["chroma_server_auth_credentials"]
API_KEY_NAME = "Authorization"

from omegaconf import OmegaConf
config = OmegaConf.load("backend/config/config.yaml")

class QueryRequest(BaseModel):
    query: str
    k: int

class HybridRetrieverAPI:

    def __init__(self):

        self.retriever = HybridRetriever(
            path_docstore=config.paths.docstores.bm25_docstore,
        )

        self.app = FastAPI(
            title="HybridRetrieverAPI"
        )

        self._initialize_routes()

    @staticmethod
    def verify_api_key(r: Request):
        api_key_input = r.headers[API_KEY_NAME]
        if api_key_input != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API Key")
        return api_key_input

    def _initialize_routes(self):

        @self.app.get("/")
        async def root():
            return {"message": "Hybrid Retriever up and running."}

        @self.app.post("/retrieve")
        async def retrieve_docs(request: QueryRequest, api_key: str = Depends(HybridRetrieverAPI.verify_api_key)):
            retrieved_documents = self.retriever.retrieve(
                query=request.query,
                k=request.k
            )
            return {
                "query": request.query,
                "res": retrieved_documents
            }

        @self.app.get("/shutdown")
        def shutdown(api_key: str = Depends(HybridRetrieverAPI.verify_api_key)):
            os.kill(os.getpid(), signal.SIGTERM)
            return Response(status_code=200, content="Server shutting down...")

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8080, log_level="info")

def main():
    api = HybridRetrieverAPI()
    api.run()

if __name__ == "__main__":
    main()