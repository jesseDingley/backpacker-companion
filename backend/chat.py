import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict

from backend.base import Base
from backend.const import CST
from backend.prompts import Prompts
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from huggingface_hub import login

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

        prompts = Prompts()

        # chain that stuffs documents into context
        combine_docs_chain = create_stuff_documents_chain(
            llm=llm, prompt=prompts.qa_chat_prompt_template
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=vectordb.as_retriever(),
            prompt=prompts.rephrase_chat_prompt_template,
        )

        self.qa = create_retrieval_chain(
            retriever=history_aware_retriever, combine_docs_chain=combine_docs_chain
        )

    def call(self, query: str, chat_history: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        Calls LLM with a query and returns generated response.

        Args:
            query (str): user query
            chat_history (List[Tuple[str, str]]): chat history like [("human": msg) -> ("ai": response) -> ...]

        Returns:
            Dict[str, str]: out dict containing query, response, and source docs
        """

        no_reply_count = 0

        while True:

            generated_response = self.qa.invoke(
                input={"input": query + " AI: ", "chat_history": chat_history}
            )

            if not generated_response["answer"].strip() in ["", "."]:
                break

            no_reply_count += 1

            if no_reply_count == 3:
                generated_response["answer"] = "I'm sorry, I don't understand."
                break

        formatted_generated_response = {
            "query": generated_response["input"],
            "result": generated_response["answer"],
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
