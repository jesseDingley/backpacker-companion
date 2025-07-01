from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from pydantic import BaseModel
from components.hybrid_retriever import HybridRetriever
import uvicorn

import logging
logging.basicConfig(
    format="%(levelname)s:  %(message)s"
)

from omegaconf import OmegaConf
config = OmegaConf.load("config/config.yaml")

class QueryRequest(BaseModel):
    query: str
    k: int
    threshold: float

class HybridRetrieverAPI:

    def __init__(self):

        self.retriever = HybridRetriever(
                path_docstore=config.paths.docstores.bm25_docstore
        )

        self.startup = True

        self.app = FastAPI(
            title="HybridRetrieverAPI"
        )

        self._register_retriever_refresher()
        self._initialize_routes()

    def _register_retriever_refresher(self):

        @self.app.on_event("startup")
        @repeat_every(seconds=60*50) #50 mins
        def get_retriever():
            if not self.startup:
                logging.warning("Refreshing Retriever...")
                self.retriever = HybridRetriever(
                    path_docstore=config.paths.docstores.bm25_docstore
                )
            self.startup = False
                

    def _initialize_routes(self):

        @self.app.get("/")
        async def root():
            return {"message": "Hybrid Retriever up and running."}

        @self.app.post("/retrieve")
        async def retrieve_docs(request: QueryRequest):
            retrieved_documents = self.retriever.retrieve(
                query=request.query,
                k=request.k,
                threshold=request.threshold,
            )
            return {
                "query": request.query,
                "res": retrieved_documents
            }

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8080, log_level="info")

def main():
    api = HybridRetrieverAPI()
    api.run()

if __name__ == "__main__":
    main()