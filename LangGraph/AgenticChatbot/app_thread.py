from agentic_chatbot_backend import chatbot
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
import uuid

# new convo er jonno unique thread id generate korbe
def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def reset_chat():

    # Generate and assign a new thread ID
    st.session_state.thread_id = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state.message_history = []

    # Add the new thread to the conversation list
    add_thread(st.session_state.thread_id)

def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])


st.title("Agentic Bot with Langgraph")

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()


if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = []


add_thread(st.session_state["thread_id"])

st.sidebar.title("My Conversations")

# current chat reset and new thread create
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()




for thread_id in st.session_state.chat_threads[::-1]:

    # Create one sidebar button for every conversation
    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):

        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages into Streamlit's required message format
        temp_messages = []



        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"

            elif isinstance(message, AIMessage):
                role = "assistant"

            else:
                continue

            temp_messages.append({
                "role": role,
                "content": message.content
            })

        st.session_state.message_history = temp_messages

        st.rerun()

for message in st.session_state.message_history:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Display the message content
        st.text(message["content"])



# Create the chat input box
user_input = st.chat_input("Type here")


if user_input:

    st.session_state.message_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)


    CONFIG = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }


    def ai_stream():

        for message_chunk, metadata in chatbot.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=CONFIG,
            stream_mode="messages"
        ):

            if isinstance(message_chunk, AIMessage):
                yield message_chunk.content


    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            ai_stream()
        )


    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )