from dotenv import load_env

from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from operator import itemgetter
import streamlit as st
import tempfile
import os
import pandas as pd


st.set_page_config(page_title="Research Paper QA Chatbot", page_icon="")
st.title("Welcome to the Research Paper QA Chatbot")

def configure_retriever(file_paths):
    docs = []

    for paths in file_paths:
        loader = PyMuPDFLoader(paths)
        # extend use korar karon flatlist paoa jay jeno 
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, 
                                                chunk_overlap=200)

    doc_chunks = text_splitter.split_documents(docs)

    embeddings_model = OllamaEmbeddings(
        model="llama3",
    )

    vectordb = Chroma.from_documents(doc_chunks, embeddings_model)

    retriever = vectordb.as_retriever()
    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token:str, **kwargs):
        self.text+=token
        self.container.markdown(self.text)

        
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"],
    accept_multiple_files=True
)


retriever = configure_retriever(uploaded_files)


llm = ChatGroq(model="openai/gpt-oss-120b", streaming=True)

qa_template = """
Use only the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know,
don't try to make up an answer. Keep the answer as concise as possible.

{context}

Question: {question}
"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)

def merging_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

qa_rag_chain = ({
    "context": 


})