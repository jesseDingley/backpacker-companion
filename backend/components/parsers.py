from pydantic import BaseModel, Field
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


class RetrievalNecessity(BaseModel):
    """
    Desired output JSON structure
    for LLM call determining whether the user query requires retrieval.
    """

    is_retrieval_needed: str = Field(
        description="'yes' or 'no' to whether the user question requires retrieving information from the vector database."
    )


class Parsers:
    """
    Available parsers
    """

    off_topic_verification_parser = JsonOutputParser(pydantic_object=IsOffTopic)
    retrieval_necessity_parser = JsonOutputParser(pydantic_object=RetrievalNecessity)


"""
CONTEXT RELEVANCE

    Options:
        1. Context is relevant to the query
            EX: query: "hostels in Paris"; ctxt: "Montmartre hostel"
            => generate response
        2. Context is empty and query is a conversational message
            EX: query: "thanks"; ctxt: ""
            => generate response
        3. Context is empty and query is a question
            EX: query: "hostels in Paris"; ctxt: ""
            => generate apology ('i lack info')
        4. Context is irrelevant to query
            EX: query: "hostels in Paris": ctxt: "la rochelle hostel"
            => generate apology ('i lack info')



"""
