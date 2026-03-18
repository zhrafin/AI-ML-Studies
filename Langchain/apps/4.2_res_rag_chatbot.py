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
        

def build_retriever(upload_files):
    #shob textt store korbo. docs = all pages from all PDFs
    docs = []

    # temporary folder/directory toiri jekhane shob files thakbe. 
    # source uploaded_files, dest temp dir
    temp_dir = tempfile.TemporaryDirectory()

    # prottek the input files er kaaj
    for file in upload_files:
        #Save each file into the folder
        file_path = os.path.join(temp_dir.name, file.name)

        ## SAVING THE FILES
        # turning streamlit files into memory
        # open() for read and write
        # "wb" = write binary file, model er jonno PDF files are not texts they are binary
        # with is a safe wayy to open files
        # we are using f to store opened file in the variable 
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

        # Reading the files
        loader = PyMuPDFLoader(file_path)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 200
    )

    doc_chunks = text_splitter.split_documents(docs)

 
    embeddings_model = OllamaEmbeddings(
        model="llama3",
    )       

    vectordb = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embeddings_model
    )

    #database theke relevent obects aina dibe
    # 
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever

        
def collect_sources(docs):
    #shob source collect kore rakhar jonno
    source = []
    # set only unique value store kore
    # automatically duplicate values ignore kore
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page =  doc.metadata.get("page", "N/A")

        key = (source, page)

        if key not in seen:
            seen.add(key)
            source.append({
                "source":source,
                "page":page,
                "content": doc.page_content[:200]

            })

    return source[:3]

    
