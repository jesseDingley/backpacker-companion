import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Iterator

from backend.base import Base
from backend.const import CST
from backend.prompts import Prompts
from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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

        callbacks = [StreamingStdOutCallbackHandler()]

        llm = HuggingFaceEndpoint(
            repo_id=CST.LLM,
            task="text-generation",
            max_new_tokens=512,
            temperature=0.1,
            callbacks=callbacks,
            streaming=True,
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

        self.formatted_output = None

    def stream(self, query: str, chat_history: List[Tuple[str, str]]) -> Iterator[str]:
        """
        Streams LLM response given a query and chat history.

        Args:
            query (str): user query
            chat_history (List[Tuple[str, str]]): chat history like [("human": msg) -> ("ai": response) -> ...]

        Returns:
            Iterable[str]: iterator on llm response chunks
        """
        self.formatted_output = {}
        acc_answer = ""
        for chunk in self.qa.stream(
            input={"input": query + " AI: ", "chat_history": chat_history}
        ):
            if input_chunk := chunk.get("input"):
                self.formatted_output["input"] = input_chunk

            if context_chunk := chunk.get("context"):
                self.formatted_output["context"] = context_chunk

            if answer_chunk := chunk.get("answer"):
                if answer_chunk == "</s>":
                    break
                acc_answer += answer_chunk
                yield answer_chunk

        self.formatted_output["answer"] = acc_answer

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

            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                st.write_stream(
                    self.stream(
                        prompt,
                        st.session_state["chat_history"],
                    )
                )

            st.session_state["chat_history"].append(("human", prompt))
            st.session_state["chat_history"].append(
                ("ai", self.formatted_output["answer"])
            )
