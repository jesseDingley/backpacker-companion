from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.const import CST
from backend.components.parsers import Parsers


class Prompts:

    def __init__(self) -> None:

        self.qa_chat_prompt_template = ChatPromptTemplate.from_template(
            CST.QA_SYS_PROMPT
        )

        self.rephrase_chat_prompt_template = ChatPromptTemplate.from_template(
            CST.REPHRASE_SYS_PROMPT
        )

        off_topic_verification_prompt_template_tmp = ChatPromptTemplate.from_template(
            CST.OFF_TOPIC_SYS_PROMPT
        )

        self.off_topic_verification_prompt_template = off_topic_verification_prompt_template_tmp.partial(
            format_instructions=Parsers.off_topic_verification_parser.get_format_instructions()
        )
