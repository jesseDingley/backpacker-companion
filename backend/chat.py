import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Iterator

from backend.base import Base
from backend.const import CST
from backend.prompts import Prompts
from backend.retriever import Retriever
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
            max_new_tokens=1024,
            temperature=0.1,
            callbacks=callbacks,
            streaming=True,
        )

        retriever = Retriever(vectordb=vectordb, k=CST.K, threshold=CST.THRESHOLD)

        prompts = Prompts()

        # chain that stuffs documents into context
        combine_docs_chain = create_stuff_documents_chain(
            llm=llm, prompt=prompts.qa_chat_prompt_template
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=prompts.rephrase_chat_prompt_template,
        )

        self.qa = create_retrieval_chain(
            retriever=history_aware_retriever, combine_docs_chain=combine_docs_chain
        )

        self.formatted_output = None
        self.ai_msg_placeholder = None

    @staticmethod
    def write_human_msg(msg: str) -> None:
        """
        Writes human message to Streamlit app

        Args:
            msg (str): message to write
        """
        st.write(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {CST.MESSAGE_ALIGNMENT};">
                    <div style="background: {CST.MESSAGE_BG_COLOR}; color: white; border-radius: 15px; padding: 10px; margin-right: 5px; max-width: 75%; font-size: 14px;">
                        {msg} \n </div>
                </div>
                """,
            unsafe_allow_html=True,
        )

    def stream(self, query: str, chat_history: List[Tuple[str, str]]) -> Iterator[str]:
        """
        Streams LLM response given a query and chat history.

        Args:
            query (str): user query
            chat_history (List[Tuple[str, str]]): chat history like [("user": msg) -> ("assistant": response) -> ...]

        Returns:
            Iterable[str]: iterator on llm response chunks
        """
        started_streaming = False
        self.formatted_output = {}
        acc_answer = ""
        for chunk in self.qa.stream(
            input={"input": query + " ASSISTANT: ", "chat_history": chat_history}
        ):
            if input_chunk := chunk.get("input"):
                self.formatted_output["input"] = input_chunk

            if context_chunk := chunk.get("context"):
                self.formatted_output["context"] = context_chunk

            if answer_chunk := chunk.get("answer"):
                if not started_streaming:
                    started_streaming = True
                    self.ai_msg_placeholder.empty()
                if answer_chunk == "</s>":
                    break
                acc_answer += answer_chunk
                yield answer_chunk

        self.formatted_output["answer"] = acc_answer

    def run_app(self, debug=True) -> None:
        """
        Run streamlit app
        """
        st.image(self.path_title_image, width=100)
        st.title(CST.NAME)
        st.caption(
            f"{CST.NAME}, your pocket backpacking companion can help you with any questions you may have about backpacking: recommendations, itineraries, safety and budgeting tips, etc."
        )

        if not "chat_history" in st.session_state:
            st.session_state["chat_history"] = [
                ("assistant", f"Hey there! {CST.NAME} here, how can I help you?")
            ]

        for role, msg in st.session_state["chat_history"]:
            if role == "user":
                Chat.write_human_msg(msg)
            else:
                with st.chat_message(role):
                    st.empty()
                    st.write(msg)

        if prompt := st.chat_input():

            Chat.write_human_msg(prompt)

            with st.chat_message("assistant"):

                self.ai_msg_placeholder = st.empty()
                self.ai_msg_placeholder.write("Hmm...")

                st.write_stream(
                    self.stream(
                        prompt,
                        st.session_state["chat_history"],
                    )
                )

            st.session_state["chat_history"].append(("user", prompt))
            st.session_state["chat_history"].append(
                ("assistant", self.formatted_output["answer"])
            )

            if debug:
                print()
                print("===========")
                print(self.formatted_output)
                print("=============")
                print()
