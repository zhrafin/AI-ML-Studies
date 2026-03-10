from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
import streamlit as st
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver  

st.set_page_config(page_title="Research Assistant", page_icon="🤖",)
st.subheader("Research Assistant - GPT OSS")

search = GoogleSerperAPIWrapper()
checkpointer = InMemorySaver()

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()

# Notun chat toiri
if "sessions" not in st.session_state:
    st.session_state.sessions = {
        "Chat 1":[]
    }

# ekhon kon chat khola ase janar jonno
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"


if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1

# current_chat er shob messages string akare nicchi
# current_chat direct nile just CHat 1 ashbe. texts ashbe  current method e

#st.session_state.sessions e thake
# st.session_state.sessions = {
#     "Chat 1": [
#         {"role": "user", "content": "Hello"},
#         {"role": "assistant", "content": "Hi"}
#     ],
#     "Chat 2": [
#         {"role": "user", "content": "Explain AI"}
#     ]
# }


current_messages = st.session_state.sessions[st.session_state.current_chat]

# Existing message print korte thakbo
for message in current_messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


# Sidebar Start
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Select model",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "llama-3.1-8b-instant",
        ], 
        index=0
    )

    enable_web_search = st.checkbox(
        "Enable Web Search",
        value=True
    )

    st.divider()
    st.subheader("Chats")

    # sessions er value gula key wise jabe jehetu ekta list
    chat_names = list(st.session_state.sessions.keys())
    
    selected_chat=st.selectbox(
            "Select Chat",
            chat_names,
            # active chat er position ber kore box e show kore
            index=chat_names.index(st.session_state.current_chat)
        )
    
    # new current chat e jaoa
    st.session_state.current_chat = selected_chat
 
    col1, col2 = st.columns(2)

    with col1:
        if st.button("New Chat"):
            st.session_state.chat_counter +=1
            new_chat_name = f"Chat {st.session_state.chat_counter}"
            st.session_state.sessions[new_chat_name] = []
            st.session_state.current_chat = new_chat_name
            #proti action er por rerun
            st.rerun()
    with col2:
        if st.button("Delete Chat"):
            if (st.session_state.chat_counter > 1):
                del st.session_state.sessions[st.session_state.current_chat]
                st.session_state.current_chat = list(st.session_state.sessions.keys())[0]
                st.rerun()


# Agent Build from here

def build_agent(model_name: str, enable_web_search: bool):
    llm = ChatGroq(model=model_name)
    
    tools = []

    if enable_web_search:
        tools=[search.run]

    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        system_prompt=
        """
            You are a research assistant.

            Use web search when needed.
            Do not paste raw search results, snippets, or repeated text.
            Do not copy tool output directly.

            Write a clean final answer in this format:

            1. Direct answer in 1 to 3 sentences
            2. Short explanation
            3. Sources:
            - Website name
            - URL

            If search results conflict, say so clearly.
            If you are unsure, say you are unsure.
        """
    )

    return agent

agent = build_agent(model_name, enable_web_search)


query = st.chat_input("Ask Your Research Assistant... ")

if query:
    st.chat_message("user").markdown(query)
    current_messages.append({"role": "user", "content": query})

    res = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": st.session_state.current_chat}}
    )

    st.chat_message("ai").markdown(res["messages"][-1].content)
    current_messages.append({"role": "ai", "content": res["messages"][-1].content})

