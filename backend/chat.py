from typing import List, Tuple, Iterator
import logging
import time
import re
import random
from requests import HTTPError
from PIL import Image

from backend.base import Base, wake_up_llm_endpoint
from backend.config.const import CST
from backend.components.prompts import Prompts, ShortInstructions
from backend.components.retriever import Retriever
from backend.components.parsers import Parsers

from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.schema.output_parser import StrOutputParser

import streamlit as st

class Chat(Base):
    """
    Class for Chat frontend
    """

    def __init__(self) -> None:
        super().__init__()

        if self.RETRIEVER == "hybrid":

            retriever = Retriever(
                search_type="hybrid", k=CST.K, threshold=CST.THRESHOLD
            )

        else:
            
            vectordb = Chroma(
                client=self.chroma_client,
                embedding_function=self.embeddings,
                collection_name=self.collection_config["NAME"]
            )

            retriever = Retriever(
                search_type="vector", vectordb=vectordb, k=CST.K, threshold=CST.THRESHOLD
            )

        prompts = Prompts()

        # chain that stuffs documents into context
        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm, prompt=prompts.qa_chat_prompt_template
        )

        self.qa = create_retrieval_chain(
            retriever=retriever, combine_docs_chain=combine_docs_chain
        )

        self.off_topic_verification_chain = (
            prompts.off_topic_verification_prompt_template
            | self.llm
            | Parsers.off_topic_verification_parser
        )

        self.retrieval_necessity_chain = (
            prompts.retrieval_necessity_prompt_template
            | self.llm
            | Parsers.retrieval_necessity_parser
        )

        self.simple_chat_chain = prompts.conversational_chat_prompt_template | self.llm | StrOutputParser()

        self.rephrase_chain = prompts.rephrase_chat_prompt_template | self.llm | StrOutputParser()

        self.assistant_icon = Image.open(self.paths["ASSISTANT_ICON"])

        with open(self.paths["SIDEBAR"], "r") as f:
            self.sidebar_content = f.read()

        self.formatted_output = None
        self.ai_msg_placeholder = None

        self.MESSAGE_ALIGNMENT = self.config.app.ui.message_alignment
        self.MESSAGE_BG_COLOR = self.config.app.ui.message_bg_color

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
    
    @staticmethod
    def limit_chat_history(chat_history: List[Tuple[str, str]], limit: int) -> List[Tuple[str, str]]:
        """
        Limits chat history to N most recent turns, 
        where a turn is a [user -> assistant] sequence of dialogue.

        Args:
            chat_history (List[Tuple[str, str]]): chat history
            limit (int): max number of turns to keep.

        Returns:
            List[Tuple[str, str]]: snipped chat history.
        """
        return chat_history[len(chat_history) - 2*limit:]
        
    def write_header(self) -> None:
        """Writes header to streamlit page."""
        st.image(self.paths["TITLE_IMAGE"], width=100)
        st.title(self.NAME)
        st.caption(
            f"{self.NAME}, your pocket backpacking companion can help you with any questions you may have about backpacking: recommendations, itineraries, safety and budgeting tips, etc."
        )
        st.caption(
            f"The first reply from {self.NAME} might take a minute or two if the server has been asleep for a while."
        )

    def write_human_msg(self, msg: str) -> None:
        """
        Writes human message to Streamlit app

        Args:
            msg (str): message to write
        """
        st.write(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {self.MESSAGE_ALIGNMENT};">
                    <div style="background: {self.MESSAGE_BG_COLOR}; color: white; border-radius: 15px; padding: 10px; margin-right: 5px; max-width: 75%; font-size: 16px;">
                        {msg} \n </div>
                </div>
                """,
            unsafe_allow_html=True,
        )

    def is_query_off_topic(
        self, query: str, chat_history: List[Tuple[str, str]]
    ) -> str:
        """
        Determines whether user query is off topic.

        Args:
            query (str): user query
            chat_history (List[Tuple[str, str]]): chat history like [("user": msg) -> ("assistant": response) -> ...]

        Returns:
            str: Refusal message if off topic.
        """

        chat_history = Chat.limit_chat_history(
            chat_history, 
            limit=CST.MAX_TURNS
        )

        try:

            output_json = self.off_topic_verification_chain.invoke(
                {"input": query, "chat_history": chat_history}
            )

            is_off_topic = output_json["is_off_topic"].lower().strip().strip("'").strip('"')

            if is_off_topic == "yes":
                return output_json["refusal_message"]

            if is_off_topic == "no":
                return None

            raise ValueError("Invalid JSON.")

        except Exception:

            logging.warning("Failed to determine if query is off topic. Defaulting to False.")
            return None

    def is_retrieval_needed(self, query: str) -> bool:
        """
        Determines whether user query requires retrieval.

        Args:
            query (str): user query

        Returns:
            bool: True if retrieval is needed, False otherwise.
        """
        try:
            output_json = self.retrieval_necessity_chain.invoke({"input": query})
            
            # Try to get the value, handling potential typos in the key name
            value = output_json.get("is_retrieval_needed") or output_json.get("is_retrival_needed")
            
            if value is None:
                logging.warning(f"Unexpected JSON structure: {output_json}. Defaulting to True.")
                return True
                
            return value.lower().strip().strip("'").strip('"') == "yes"
        except Exception as e:
            logging.warning(f"Failed to determine retrieval necessity: {e}. Defaulting to True.")
            return True

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

    def stream(self, query: str, rephrased_query: str, chat_history: List[Tuple[str, str]], id_token: str) -> Iterator[str]:
        """
        Streams LLM response given a query and chat history.

        Args:
            query (str): user query
            rephrased_query (str): rephrased user query based on history for retriever.
            chat_history (List[Tuple[str, str]]): chat history like [("user": msg) -> ("assistant": response) -> ...]
            id_token (str): Google ID Token obtained from user login.

        Returns:
            Iterable[str]: iterator on llm response chunks
        """
        chat_history = Chat.limit_chat_history(
            chat_history, 
            limit=CST.MAX_TURNS
        )
        started_streaming = False
        starts_with_ai = False
        self.formatted_output = {}
        acc_answer = ""

        def should_skip_chunk_start(chunk: str) -> bool:
            """
            Helper function to handle AI prefix removal and stream start detection.
            Returns True if chunk should be skipped, False otherwise.
            Updates nonlocal variables started_streaming and starts_with_ai.
            """
            nonlocal started_streaming, starts_with_ai
            
            if not started_streaming:
                # remove "AI: " from start of generation
                if chunk.strip() in ["", "\n"]:
                    return True
                if chunk.strip() == "AI":
                    starts_with_ai = True
                    return True
                if starts_with_ai and chunk.strip() == ":":
                    starts_with_ai = False
                    return True

                started_streaming = True
                self.ai_msg_placeholder.empty()

            return False

        def process_chunk(chunk: str) -> str:
            """
            Process chunk with vocabulary diversification and dollar sign escaping.
            Returns the processed chunk.
            """
            # diversify vocab
            chunk = Chat.diversify_vocabulary(chunk)

            # escape dollar sign for markdown
            chunk = Chat.esc_dollar_sign(chunk)

            return chunk

        # Check if retrieval is needed using the REPHRASED query
        if not self.is_retrieval_needed(rephrased_query):

            logging.info("Skipping retrieval.")
            
            # Use simple chain without retrieval
            for chunk in self.simple_chat_chain.stream(
                input={
                    "input": query, 
                    "chat_history": Chat.format_chat_history(chat_history),
                }
            ):

                if should_skip_chunk_start(chunk):
                    continue

                # end stream
                if chunk == "</s>":
                    break

                chunk = process_chunk(chunk)

                acc_answer += chunk
                yield chunk

            self.formatted_output["answer"] = acc_answer
            return

        try:

            for chunk in self.qa.stream(
                input={
                    "input": query, 
                    "rephrased_input": rephrased_query, 
                    "chat_history": Chat.format_chat_history(chat_history),
                    "id_token": id_token,
                }
            ):

                if input_chunk := chunk.get("input"):
                    self.formatted_output["input"] = input_chunk

                if context_chunk := chunk.get("context"):
                    self.formatted_output["context"] = context_chunk

                if answer_chunk := chunk.get("answer"):
                    
                    if should_skip_chunk_start(answer_chunk):
                        continue

                    # end stream
                    if answer_chunk == "</s>":
                        break

                    answer_chunk = process_chunk(answer_chunk)

                    acc_answer += answer_chunk
                    yield answer_chunk

        except HTTPError as e:

            if '403' in str(e) or '401' in str(e):
                st.logout()
            else:
                raise

        if "context" in self.formatted_output:

            top_ranking_documents = []
            used = []
            for doc in self.formatted_output["context"]:
                if doc.page_content == ShortInstructions.no_docs_found_response:
                    break
                if not doc.metadata["link"] in used:
                    if "score" in doc.metadata:
                        if (doc.metadata["score"] <= CST.THRESHOLD):
                            continue
                    top_ranking_documents.append(doc)
                    used.append(doc.metadata["link"])

            if top_ranking_documents == []:
                self.formatted_output["answer"] = acc_answer
                return

            top_ranking_documents = top_ranking_documents[:3]

            context_text = f"\n\n For more information, check out the following:\n\n"
            self.formatted_output["source"] = context_text
            for token in context_text.split(" "):
                time.sleep(0.01)
                yield " "
                time.sleep(0.01)
                yield token

            for doc in top_ranking_documents:
                url = doc.metadata["link"]
                title = doc.metadata["title"]
                doc_context_text = f"- [{title}]({url})\n"
                for token in doc_context_text.split(" "):
                    time.sleep(0.01)
                    yield " "
                    time.sleep(0.01)
                    yield token
                self.formatted_output["source"] += doc_context_text

        self.formatted_output["answer"] = acc_answer

    def run_app(self) -> None:
        """
        Run streamlit app
        """

        if "num_interactions" not in st.session_state: 
            st.session_state["num_interactions"] = 0

        st.session_state["num_interactions"] += 1

        if not st.user.is_logged_in:

            logging.info("User is logged out.")

            # Header
            self.write_header()

            # Log user in with Google
            if st.button(
                "![](/backend/images/google-icon.png) Sign in with Google", 
                icon=":material/login:", 
                type="primary", 
                use_container_width=True
            ):
                st.login("google")

        else:

            logging.info("User is logged in.")
            
            if st.session_state["num_interactions"] == 1:
                logging.info("Refreshed page to prevent ghosting.")
                st.rerun()

            # Header
            self.write_header()

            # Sidebar 
            with st.sidebar:
                if st.button("New Chat", use_container_width=True, type="primary"):
                    for key in st.session_state.keys():
                        if key != "num_interactions":
                            del st.session_state[key]

                if st.button("Sign out", use_container_width=True):
                    st.logout()

                st.markdown(self.sidebar_content)

            # Init chat history
            if not "chat_history" in st.session_state:
                st.session_state["chat_history"] = [
                    ("assistant", f"Hey there {st.user['name'].split(' ')[0]}! {self.NAME} here, how can I help you?")
                ]

            # Init data sources to display on each assistant message
            if not "references" in st.session_state:
                st.session_state["references"] = [""]

            # Display whole chat history
            for role_msg_tuple, reference in zip(
                st.session_state["chat_history"], st.session_state["references"]
            ):
                role = role_msg_tuple[0]
                msg = role_msg_tuple[1]
                if role == "user":
                    self.write_human_msg(msg)
                else:
                    with st.chat_message(role, avatar=self.assistant_icon):
                        st.empty()
                        st.write(msg + reference)

            # New message from User
            if prompt := st.chat_input():

                self.write_human_msg(prompt)

                # Generate and stream response from Assistant
                with st.chat_message("assistant", avatar=self.assistant_icon):

                    self.ai_msg_placeholder = st.empty()
                    self.ai_msg_placeholder.write("Hmm...")

                    # 1. Check if off-topic (using raw input, but prompt has history)
                    refusal_message = self.is_query_off_topic(
                        prompt, st.session_state["chat_history"]
                    )

                    if self.debug:
                        print('\n\n')
                        if refusal_message is not None:
                            logging.info("Is off topic: True")
                            logging.info(f"Refusal message: <<<{refusal_message}>>>")
                        else:
                            logging.info("Is off topic: False")
                        print('\n')

                    if refusal_message is not None:

                        st.write_stream(self.stream_refusal_message(refusal_message))

                    else:

                        # 2. Rephrase only if on-topic
                        rephrased_input = self.rephrase_chain.invoke({
                            "input": prompt, 
                            "chat_history": Chat.format_chat_history(st.session_state["chat_history"])
                        })

                        if self.debug:
                            print("\n\n")   
                            logging.info(f"Rephrased Input: {rephrased_input}\n")

                        st.write_stream(
                            self.stream(
                                prompt,
                                rephrased_input,
                                st.session_state["chat_history"],
                                st.user["id_token"],
                            )
                        )

                # Update chat history
                st.session_state["chat_history"].append(("user", prompt))
                st.session_state["chat_history"].append(("assistant", self.formatted_output["answer"]))

                st.session_state["references"].append("")
                if "source" in self.formatted_output:
                    st.session_state["references"].append(self.formatted_output["source"])
                else:
                    st.session_state["references"].append("")

                # Debug statements
                if self.debug:
                    if "context" in self.formatted_output:
                        print("\n\n")
                        print("=============")
                        for i, doc in enumerate(self.formatted_output["context"]):
                            print(f"DOCUMENT {i}: ")

                            print("<---")
                            print(doc)
                            print("--->")
                        print("=============")
                        print("\n\n")
                    else:
                        print("\n\nNO DOCUMENTS RETRIEVED")

    def run_app_safe(self) -> None:
        """
        Run streamlit app

        with a try except to deal with LLM Endpoint auto-scaling to zero
        """
        try:
            self.run_app()
        except Exception as e:
            if "503" in str(e) or "service_unavailable" in str(e).lower():
                logging.warning("LLM server down. Starting it back up.")
                wake_up_llm_endpoint.clear()
                wake_up_llm_endpoint()
                self.run_app()
            else:
                raise
