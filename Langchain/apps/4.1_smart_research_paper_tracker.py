from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import GoogleSerperAPIWrapper
import streamlit as st

st.subheader("Track Your Research Paper")

enable_web_search = st.checkbox(
        "Enable Web Search",
        value=True
    )

st.write("Web search enabled:", enable_web_search)

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)



db = SQLDatabase.from_uri("sqlite:///my_papers.db")

db.run("""






""")

llm = ChatGroq(model="openai/gpt-oss-20b")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

search = GoogleSerperAPIWrapper()

system_prompt ="""

    You are an intelligent research paper tracker, 
    who tracks papers from your database and 
    and when asked you search for papers in google too.

"""

def build_agent(enable_web_search: bool):
    if enable_web_search:
        agent = create_agent(
            model = llm,
            tools = [search.run, tools],
            checkpointer = st.session_state.memory,
            system_prompt=system_prompt,
        )
    else:
        agent = create_agent(
            model = llm,
            tools = tools,
            checkpointer = st.session_state.memory,
            system_prompt=system_prompt,
        )
    return agent


agent = build_agent( enable_web_search)


query = st.chat_input("Ask Your Research Assistant... ")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    res = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": st.session_state.current_chat}}
    )

    st.chat_message("ai").markdown(res["messages"][-1].content)
    st.session_state.messages.append({"role": "ai", "content": res["messages"][-1].content})
        
    

