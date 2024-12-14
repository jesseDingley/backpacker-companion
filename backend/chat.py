from typing import List, Tuple, Dict, Iterator
import time
import re
import random

from backend.base import Base
from backend.const import CST
from backend.components.prompts import Prompts
from backend.components.retriever import Retriever
from backend.components.parsers import Parsers
from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from huggingface_hub import login
from PIL import Image

import streamlit as st


class Chat(Base):
    """
    Class for Chat frontend
    """

    def __init__(self) -> None:
        super().__init__()

        vectordb = Chroma(
            client=self.chroma_client,
            embedding_function=self.embeddings,
            collection_name=CST.COLLECTION
        )

        login(token=st.secrets["HUGGINGFACE_API_KEY"])

        callbacks = [StreamingStdOutCallbackHandler()]

        llm = HuggingFaceEndpoint(
            repo_id=CST.LLM,
            task="text-generation",
            max_new_tokens=CST.MAX_NEW_TOKENS,
            temperature=CST.TEMPERATURE,
            callbacks=callbacks,
            streaming=True,
            stop_sequences=["<unk>", "</s>", "}"],
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

        self.off_topic_verification_chain = (
            prompts.off_topic_verification_prompt_template
            | llm
            | Parsers.off_topic_verification_parser
        )

        self.assistant_icon = Image.open(self.path_assistant_icon)

        with open(self.path_sidebar_md, "r") as f:
            self.sidebar_content = f.read()

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

    @staticmethod
    def diversify_vocabulary(current_chunk: str) -> str:
        """
        Mixes up the vocabulary for overused terms like 'bloody', 'mate'

        Args:
            current_chunk (str): current chunk

        Returns:
            str: possible other word / chunk
        """
        if current_chunk == " mate":
            return random.choice(
                [" mate", " my friend", " bud", " amigo", " pal", " ", " partner"]
            )
        if current_chunk == " bloody":
            return random.choice(
                [
                    " bloody",
                    " fucking",
                    " absolutely",
                    " damn",
                    " just",
                    " hella",
                    " totally",
                ]
            )
        return current_chunk
    
    @staticmethod
    def format_chat_history(chat_history: List[Tuple[str, str]]) -> str:
        """Formats chat history into a string for prompt templates.
        
        Example result:
            "Assistant: Hey there! Alma here, how can I help you?
            User: How expensive is Thailand to travel to?
            Assistant: Thailand is a backpacker's dream! It's cheap as chips."

        Args:
            chat_history (List[Tuple[str, str]]): chat history like [("user": msg) -> ("assistant": response) -> ...]

        Returns:
            str: formatted chat history.
        """
        return "\n".join(
            f"{role.capitalize()}: {content}" for role, content in chat_history
        )

    @staticmethod
    def esc_dollar_sign(current_chunk: str) -> str:
        """
        Escapes dollar sign to prevent bad formatting in markdown interface output.

        Args:
            current_chunk (str): current chunk

        Returns:
            str: formatted chunk
        """
        return re.sub(r"(?<!\\)\$", "\$", current_chunk)

    def is_query_off_topic(
        self, query: str, chat_history: List[Tuple[str, str]], acc: int = 0
    ) -> str:
        """
        Determines whether user query is off topic.

        Args:
            query (str): user query
            chat_history (List[Tuple[str, str]]): chat history like [("user": msg) -> ("assistant": response) -> ...]

        Returns:
            str: Refusal message if off topic.
        """

        try:

            output_json = self.off_topic_verification_chain.invoke(
                {"input": query + " JSON: ", "chat_history": chat_history}
            )

            if output_json["is_off_topic"].lower().strip() == "yes":
                return output_json["refusal_message"]

            if output_json["is_off_topic"].lower().strip() == "no":
                return None

            raise ValueError("Invalid JSON.")

        except:

            if acc == 3:
                return None
            return self.is_query_off_topic(query, chat_history, acc + 1)

    def stream_refusal_message(self, refusal_message: str) -> Iterator[str]:
        """
        Streams refusal response after a user query that is off topic.

        Args:
            refusal_message (str): refusal message

        Returns:
            Iterator[str]: iterator on refusal message.
        """
        started_streaming = False
        self.formatted_output = {}
        for word in refusal_message.split(" "):
            if not started_streaming:
                started_streaming = True
                self.ai_msg_placeholder.empty()
            yield word + " "
            time.sleep(0.02)

        self.formatted_output["answer"] = refusal_message

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
        starts_with_ai = False
        self.formatted_output = {}
        acc_answer = ""
        for chunk in self.qa.stream(
            input={"input": query, "chat_history": Chat.format_chat_history(chat_history)}
        ):
            if input_chunk := chunk.get("input"):
                self.formatted_output["input"] = input_chunk

            if context_chunk := chunk.get("context"):
                self.formatted_output["context"] = context_chunk

            if answer_chunk := chunk.get("answer"):

                if not started_streaming:

                    # remove "AI: " from start of generation
                    if answer_chunk.strip() in ["", "\n"]:
                        continue
                    if answer_chunk.strip() == "AI":
                        starts_with_ai = True
                        continue
                    if starts_with_ai and answer_chunk.strip() == ":":
                        starts_with_ai = False
                        continue

                    started_streaming = True
                    self.ai_msg_placeholder.empty()

                # end stream
                if answer_chunk == "</s>":
                    break

                # diversify vocab
                answer_chunk = Chat.diversify_vocabulary(answer_chunk)

                # escape dollar sign for markdown
                answer_chunk = Chat.esc_dollar_sign(answer_chunk)

                acc_answer += answer_chunk
                yield answer_chunk

        if "context" in self.formatted_output:
            top_ranking_document = self.formatted_output["context"][0]
            if top_ranking_document.metadata["score"] < 0.8:
                url = top_ranking_document.metadata["link"]
                title = top_ranking_document.metadata["title"]
                context_text = f"\n\n For more information, check out [{title}]({url})"
                for token in context_text.split(" "):
                    time.sleep(0.01)
                    yield " "
                    time.sleep(0.01)
                    yield token
                self.formatted_output["source"] = context_text

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

        with st.sidebar:
            if st.button("New Chat", use_container_width=True, type="primary"):
                for key in st.session_state.keys():
                    del st.session_state[key]
            st.markdown(self.sidebar_content)

        if not "chat_history" in st.session_state:
            st.session_state["chat_history"] = [
                ("assistant", f"Hey there! {CST.NAME} here, how can I help you?")
            ]

        if not "references" in st.session_state:
            st.session_state["references"] = [""]

        for role_msg_tuple, reference in zip(
            st.session_state["chat_history"], st.session_state["references"]
        ):
            role = role_msg_tuple[0]
            msg = role_msg_tuple[1]
            if role == "user":
                Chat.write_human_msg(msg)
            else:
                with st.chat_message(role, avatar=self.assistant_icon):
                    st.empty()
                    st.write(msg + reference)

        if prompt := st.chat_input():

            Chat.write_human_msg(prompt)

            with st.chat_message("assistant", avatar=self.assistant_icon):

                self.ai_msg_placeholder = st.empty()
                self.ai_msg_placeholder.write("Hmm...")

                refusal_message = self.is_query_off_topic(
                    prompt, st.session_state["chat_history"]
                )

                if debug:
                    print("\nSTART\n")

                if refusal_message is not None:

                    st.write_stream(self.stream_refusal_message(refusal_message))

                else:

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
            st.session_state["references"].append("")
            if "source" in self.formatted_output:
                st.session_state["references"].append(self.formatted_output["source"])
            else:
                st.session_state["references"].append("")

            if debug:
                print()
                print("===========")
                print(self.formatted_output)
                print("=============")
                print()
