from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  
import streamlit as st


st.subheader("Taskbot - OSS")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()

if "messages" not in st.session_state:
     st.session_state.messages = []


def process_docs(path):
    # RAG PIPELINE
    ## data load
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    ## Data Split
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splitted_docs = splitter.split_documents(docs)

    # EMbedding & vectorDB
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = InMemoryVectorStore.from_documents(
        documents = splitted_docs,
        embedding = embeddings,

    )

    # Agent Creation 

    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def ret_tool(query:str):
        """
            This tool can help retrieve data from thee PDF docs.
        """

        docs = vector_store.similarity_search(query=query,  k=4)
        
        context = ""

        for doc in docs:
            context = doc.page_content + "\n"
        
        return context


    System_Prompt = """
        You are a helpful assistant that answers questions using retrieved context.
        ALWAYS use the `ret_tool` tool for questions requiring external knowledge.
    """

    agent = create_agent(
        model = llm,
        tools = [ret_tool],
        system_prompt = System_Prompt,
        checkpointer = st.session_state.memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True




#upload ui
if st.session_state.document_uploaded == False:
    uploaded = st.file_uploader(
        label="select your pdf files", 
        type=["pdf"],
        accept_multiple_files=True
        )
    if uploaded:
        with st.spinner("Processing..."):
            path ="./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_docs(path)
            st.rerun()

#chat ui

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)


    query = st.chat_input("Ask anything related to uploaded documents..")
    if query:
        st.session_state.messages.append({"role": "user", "content":query})

        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages": [{"role":"user", "content": query}]},
            {"configurable":{"thread_id": 1}}
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role": "ai", "content":answer})