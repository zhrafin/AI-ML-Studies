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


if upload_files:
    st.session_state.retriever = build_retriever(upload_files)
else:
    st.info("Upload PDF files from the sidebar.")
    st.stop()

@tool
def search_papers(question: str) -> str:
    """
    Search the uploaded PDF paper and return relevant context

    Args: 
        question: User question about the uploaded papers 
    """

    retriever = st.session_state.retriever

    if retriever is None:
        return "No PDFs are uploaded"
    
    docs = retriever.invoke(question)
    st.session_state.last_sources = collect_sources(docs)

    context = "\n\n".join(doc.page_content for doc in docs)

    if not context.strip():
        return "No relevant context was found in the uploaded papers."

    return context



llm = ChatGroq(
    model=model_name, 
    streaming=True
    )

system_prompt = """
You are a research paper question answering assistant.

Rules:
1. The user asks questions about uploaded PDF research papers.
2. Always use the search_papers tool first when the user asks about paper content.
3. Use only the retrieved context to answer.
4. If the context does not contain the answer, say you do not know.
5. Do not invent facts.
6. Be clear and concise.
7. If helpful, mention that sources are shown below the answer.
"""

def build_agent():
    agent_tools = [search_papers]

    agent = create_agent(
        model=llm,
        tools=agent_tools,
        checkpointer=st.session_state.memory,
        system_prompt=system_prompt,
    )
    return agent

agent = build_agent()


query = st.chat_input("Ask your research assistant...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        stream_box = st.empty()
        stream_handler = StreamHandler(stream_box)

        res = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {
                "configurable": {"thread_id": "single_chat"},
                "callbacks": [stream_handler]
            }
        )

        final_answer = res["messages"][-1].content

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer
        })
