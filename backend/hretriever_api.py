from typing import Dict
from fastapi import FastAPI, Header, HTTPException
from fastapi_utils.tasks import repeat_every
from pydantic import BaseModel
from components.hybrid_retriever import HybridRetriever
import uvicorn
import os
from dotenv import load_dotenv
from google.oauth2 import id_token
from google.auth.transport import requests
from datetime import timedelta, datetime, timezone
from jose import jwt, JWTError, ExpiredSignatureError

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

class UserDataRequest(BaseModel):
    user_data: Dict[str, str]

class HybridRetrieverAPI:

    def __init__(self):

        load_dotenv()

        self.jwt_algorithm = "HS256"
        self.jwt_expiry_days = 2
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY")

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


        @self.app.post("/login")
        async def login(request: UserDataRequest, authorization: str = Header(...)) -> str:

            # STEP 1: Validate Google ID Token
            token = authorization.split("Bearer ")[1]
            if not HybridRetrieverAPI.verify_google_id_token(token):
                raise HTTPException(status_code=401, detail="Invalid Authorization header")
            
            # STEP 2: Create JWT Token
            jwt_token = self.create_jwt_token(request.user_data)

            # STEP 3: Sanity Check
            self.verify_jwt_token(jwt_token)

            return jwt_token

        @self.app.post("/retrieve")
        async def retrieve_docs(request: QueryRequest, authorization: str = Header(...)) -> dict:

            token = authorization.split("Bearer ")[1]
            self.verify_jwt_token(jwt_token=token)

            retrieved_documents = self.retriever.retrieve(
                query=request.query,
                k=request.k,
                threshold=request.threshold,
            )
            return {
                "query": request.query,
                "res": retrieved_documents
            }
        
    @staticmethod
    def verify_google_id_token(token: str) -> bool:
        """Verifies ID Token from Google sign-in."""

        load_dotenv()
        WEB_CLIENT_ID = os.getenv("WEB_CLIENT_ID")

        try:
            id_token.verify_oauth2_token(token, requests.Request(), WEB_CLIENT_ID)
        except ValueError:
            # Invalid token
            return False
        return True
    
    def create_jwt_token(self, user_data: Dict[str, str]) -> str:
        """
        Generatess JWT token from user data, with an expiry of now + N days.

        Args:
            user_data (dict): User data dict where keys are 'sub', 'email', 'name'

        Returns: 
        str:  Unique JWT token
        """
        to_encode = user_data.copy()
        
        # expire time of the token
        expire = datetime.now(timezone.utc) + timedelta(days=self.jwt_expiry_days)
        to_encode.update({"exp": expire})

        # encode
        encoded_jwt = jwt.encode(
            to_encode, 
            self.jwt_secret_key, 
            algorithm=self.jwt_algorithm
        )
        
        # return the generated token
        return encoded_jwt
    
    def verify_jwt_token(self, jwt_token: str) -> Dict[str, str]:
        """
        Verifies JWT token.

        Args:
            jwt_token (str): JWT token

        Returns:
            dict: Decoded token
        """
        try:
            # try to decode the token, it will 
            # raise error if the token is not correct
            payload = jwt.decode(jwt_token, self.jwt_secret_key, algorithms=[self.jwt_algorithm])
            return payload
        except ExpiredSignatureError:
            raise HTTPException(status_code=403, detail="Expired Credentials.")
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid Credentials.")

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8080, log_level="info")

def main():
    api = HybridRetrieverAPI()
    api.run()

if __name__ == "__main__":
    main()