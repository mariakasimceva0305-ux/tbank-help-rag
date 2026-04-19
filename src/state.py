from typing import Annotated, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    documents: NotRequired[list[Document]]
    sources: NotRequired[list[str]]
    error: NotRequired[str | None]

