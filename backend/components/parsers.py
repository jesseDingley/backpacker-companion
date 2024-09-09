from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser


class IsOffTopic(BaseModel):
    """
    Desired output JSON structure
    for LLM call determining whether the user query is off topic.
    """

    is_off_topic: str = Field(
        description="'yes' or 'no' to whether the user question is off topic or not. "
    )

    refusal_message: str = Field(
        description="A message of refusal to answer the user question (leave blank if the user question is not off topic)."
    )


class Parsers:
    """
    Available parsers
    """

    off_topic_verification_parser = JsonOutputParser(pydantic_object=IsOffTopic)
