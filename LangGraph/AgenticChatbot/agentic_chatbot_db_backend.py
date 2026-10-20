from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Optional, Annotated, Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator
import sqlite3

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

class ChatState(BaseModel):
    messages: Annotated[list[BaseModel], add_messages]

def chat_node(state: ChatState):
    response = model.invoke(state.messages)

    return{
        'messages': [response]
    }

# sqlite by default multi threading allow kore na
conn = sqlite3.connect(database="chatbot.db" , check_same_thread=False) 
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

# nodes
graph.add_node('chat_node', chat_node)

# edge
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(
    checkpointer = checkpoint
)

def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None): # none karon kono specific threads chai na
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)