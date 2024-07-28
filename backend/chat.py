import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict

from backend.base import Base
from backend.const import CST
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain import hub
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from huggingface_hub import login
from langchain_core.messages import AIMessage, HumanMessage

import streamlit as st


class Chat(Base):
    """
    Class for Chat frontend
    """

    def __init__(self) -> None:
        super().__init__()
        load_dotenv()

        assert os.path.exists(self.path_vectordb), "Vector DB not found."

        vectordb = Chroma(
            persist_directory=self.path_vectordb, embedding_function=self.embeddings
        )

        login(token=os.environ["HUGGINGFACE_API_KEY"])

        llm = HuggingFaceEndpoint(
            repo_id=CST.LLM, task="text-generation", max_new_tokens=512, temperature=0.1
        )

        retrieval_qa_chat_prompt = hub.pull(CST.QA_PROMPT)
        rephrase_prompt = hub.pull(CST.REPHRASE_PROMPT)

        # chain that stuffs documents into context
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

        history_aware_retriever = create_history_aware_retriever(
            llm=llm, retriever=vectordb.as_retriever(), prompt=rephrase_prompt
        )

        self.qa = create_retrieval_chain(
            retriever=history_aware_retriever, combine_docs_chain=combine_docs_chain
        )

    @staticmethod
    def catch_empty_response(generated_response: str) -> str:
        """Catch empty response from LLM."""
        if generated_response.strip() in ["", "."]:
            return "I'm sorry, I don't understand."
        return generated_response

    def call(self, query: str, chat_history: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        Calls LLM with a query and returns generated response.

        Args:
            query (str): user query
            chat_history (List[Tuple[str, str]]): chat history like [("human": msg) -> ("ai": response) -> ...]

        Returns:
            Dict[str, str]: out dict containing query, response, and source docs
        """
        generated_response = self.qa.invoke(
            input={"input": query, "chat_history": chat_history}
        )

        formatted_generated_response = {
            "query": generated_response["input"],
            "result": Chat.catch_empty_response(generated_response["answer"]),
            "source_documents": generated_response["context"],
        }

        return formatted_generated_response

    def run_app(self) -> None:
        """
        Run streamlit app
        """
        st.title("Backpacker Companion")
        st.caption("Description")

        if not "chat_history" in st.session_state:
            st.session_state["chat_history"] = [("ai", "How can I help you?")]

        for role, msg in st.session_state["chat_history"]:
            st.chat_message(role).write(msg)

        if prompt := st.chat_input():

            st.chat_message("user").write(prompt)

            with st.spinner("Generating response . . . "):
                generated_response = self.call(
                    prompt,
                    st.session_state["chat_history"],
                )

            st.chat_message("assistant").write(generated_response["result"])

            st.session_state["chat_history"].append(("human", prompt))
            st.session_state["chat_history"].append(
                ("ai", generated_response["result"])
            )
