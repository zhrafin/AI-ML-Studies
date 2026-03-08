from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver  


st.subheader("Ask GPT-OSS with Search Tool")
st.markdown("Test GPT-OSS-20b")

llm = ChatGroq(model="openai/gpt-oss-20b", streaming="True")
search = GoogleSerperAPIWrapper()
tools = [search.run]


# ami jodi normal memory variable er moddhe rakhi tahole dekhga jabe proti new question e memory address change hobe
# thats why created st.session_state er memory variable 
# if make sure korbe j ekbar object toiri houar por memory object abar na toiri hoy

if "memory" not in st.session_state: 
    st.session_state.memory = InMemorySaver()


agent = create_agent(
    model=llm, 
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are an agent and You can search for any questions on Google"
)

query = st.chat_input("Ask anything:")


# If "messages" does not exist, this line creates it and sets it to an empty list.
#Prev chat dhore rakhar jonno use hocche
if "messages" not in st.session_state:
    st.session_state.messages = []

#Existing message print kore
for message in st.session_state.messages:
    role = message["role"] # jei role sheta define korte hobe. ekhane user
    content = message["content"] # user query
    # creates a chat bubble for the sender.
    st.chat_message(role).markdown(content)




if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    res = agent.stream(
        {"messages":[{"role": "user", "content": query}]}, 
        {"configurable": {"thread_id": "rafin420"}},
        # llm er message stream er jonno
        stream_mode="messages"
        )
    
    res_container = st.chat_message("ai")

    # ekta ai space container reference toiri hocche. Put all UI elements inside the AI chat bubble container.
    with res_container:
        # This creates an empty placeholder inside the AI message bubble.
        space=st.empty()
        
        # extra variable anar karon response k chunk e bhenge then space e bhengge bhenge dekhano 
        # Anything inside the block will appear inside that message bubble.
        message = ""

        for chunk in res:
            message = message + chunk[0].content
            space.write(message)
    
        st.session_state.messages.append({"role":"ai", "content": message})
