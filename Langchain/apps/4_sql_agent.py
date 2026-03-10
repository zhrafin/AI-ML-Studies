from dotenv import load_dotenv
load_dotenv()

### ja ja lagbe: db, llm, tools, create_agent, system_prompt

from langchain_groq import ChatGroq
from langchain_community.utilities.sql_database import SQLDatabase #db er shathe connect er jonno
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit #tools use korte dibe


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

print("hi")


