from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
import streamlit as st

st.title("Ask GPT-OSS with Search Tool")
st.markdown("Test GPT-OSS-20b")

llm = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=llm, 
    tools=[search.run],
    system_prompt="You are an agent and You can search for any questions on Google"
)

# If "messages" does not exist, this line creates it and sets it to an empty list.
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"] # jei role sheta define korte hobe. ekhane user
    content = message["content"] # user query
    # creates a chat bubble for the sender.
    st.chat_message(role).markdown(content)


query = st.chat_input("Ask anything:")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    res = agent.invoke({"messages":[{"role": "user", "content": query}]})

    st.chat_message("ai").markdown(res["messages"][-1].content)
    st.session_state.messages.append({"role":"ai", "content": res["messages"][-1].content})



