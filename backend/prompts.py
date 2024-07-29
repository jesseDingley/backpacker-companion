from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.const import CST


class Prompts:

    def __init__(self) -> None:
        self.qa_chat_prompt_template = Prompts.create_chat_prompt_template(
            CST.QA_SYS_PROMPT
        )
        self.rephrase_chat_prompt_template = Prompts.create_chat_prompt_template(
            CST.REPHRASE_SYS_PROMPT
        )

    @staticmethod
    def create_chat_prompt_template(system_prompt: str) -> ChatPromptTemplate:
        """Creates chat prompt template."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
