from dotenv import load_dotenv
load_dotenv()

### ja ja lagbe: db, llm, tools, memory, create_agent, system_prompt

from langchain_groq import ChatGroq
from langchain_community.utilities.sql_database import SQLDatabase #db er shathe connect er jonno
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit #tools use korte dibe
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.agents import create_agent
import streamlit as st

st.subheader("Taskbot - OSS")

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()

if "messages" not in st.session_state:
     st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


# sqlite is a prebuild python database
db = SQLDatabase.from_uri("sqlite:///my_tasks.db")
db.run(""" 
       CREATE TABLE IF NOT EXISTS tasks (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       title TEXT NOT NULL,
       description TEXT,
       status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
       );

""")


llm = ChatGroq(model="openai/gpt-oss-20b")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser.

CRUD OPERATIONS:
       CREATE: INSERT INTO tasks(title, description, status)
       READ: SELECT * FROM tasks WHERE ... LIMIT 10
       UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
       DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""


# st.chache_resource make sure korbe pura code rerun hoileo bar bar agent toiri jeno na hoy 
# ebong bar bar memory te database na ashe


def build_agent():
    agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt=system_prompt
)
    return agent


agent = build_agent()

query = st.chat_input("Ask me to manage your tasks? ")

if query:
      st.chat_message("user").markdown(query)
      st.session_state.messages.append({"role": "user", "content": query})
      with st.chat_message("ai"):
           with st.spinner("Processing..."):
              res=agent.invoke(
                     {"messages": [{"role":"user", "content":query}]},
                     {"configurable": {"thread_id": "rafinTaskBot"}}
               )
              
              st.markdown(res["messages"][-1].content)
              st.session_state.messages.append({"role":"ai", "content":res["messages"][-1].content})

