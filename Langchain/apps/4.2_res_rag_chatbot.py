from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import tempfile
import os
import pandas as pd

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler

st.set_page_config(page_title="File QA Chatbot", page_icon="🤖")
st.title("File QA RAG Chatbot")


if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ekahen amar shob pdf thakbe, refresh e pdf haray jabe na
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

for message in st.session_state.messages:
    st.chat_message(message("role")).markdown(message("content"))

# Sidebar creation
with st.sidebar:
    st.subheader("Upload and Seletct")

    model_name=st.selectbox(
        "Select Model",
        ["openai/gpt-oss-20",
         "openai/gpt-oss-120b",
         "llama-3.1-8b-instant"],
         index=0
    )

    upload_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def new_tokens(self, token, **kwargs):
        self.text+=token
        self.container.markdown(self.text)
        
