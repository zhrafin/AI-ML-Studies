from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
import streamlit as st
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver  

st.subheader("Research Assistant - GPT OSS")

llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
search = GoogleSerperAPIWrapper()
checkpointer = InMemorySaver()

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()


if "messeges" not in st.session_state:
    st.session_state.messeges = []


for messege in st.session_state.messeges:
    role = messege["role"]
    content = messege["content"]
    st.chat_message(role).markdown(content)



agent = create_agent(
    model=llm, 
    tools=[search.run],
    checkpointer=checkpointer,
    system_prompt="You are Expert Research Assistant who searches in the web For valid source and answers from the sources with proof (eg. link of the websites, name of the website) attached to the answer."
)

query = st.chat_input("Ask Your Research Assistant... ")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messeges.append({"role": "user", "content": query})

    res = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "rafinAI"}}
    )

    st.chat_message("ai").markdown(res["messages"][-1].content)
    st.session_state.messeges.append({"role": "ai", "content": res["messages"][-1].content})
