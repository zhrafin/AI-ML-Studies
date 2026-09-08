from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Optional, Annotated, Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

class ChatState(BaseModel):
    messages: Annotated[list[BaseModel], add_messages]

def chat_node(state: ChatState):
    response = model.invoke(state.messages)

    return{
        'messages': [response]
    }

checkpoint = MemorySaver()

graph = StateGraph(ChatState)

# nodes
graph.add_node('chat_node', chat_node)

# edge
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(
    checkpointer = checkpoint
)

