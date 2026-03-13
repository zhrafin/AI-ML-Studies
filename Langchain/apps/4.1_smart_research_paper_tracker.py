from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import GoogleSerperAPIWrapper
import streamlit as st
from langchain.tools import tool

st.subheader("Track Your Research Paper")

enable_web_search = st.checkbox(
        "Enable Web Search",
        value=True
    )

st.write("Web search enabled:", enable_web_search)

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()

# joto khon user confrim korbe na totokhon ekta static response er jonno 
if "pending_paper_matches" not in st.session_state:
    st.session_state.pending_paper_matches = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)



db = SQLDatabase.from_uri("sqlite:///my_papers.db")

db.run("""
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL UNIQUE,
    link TEXT NOT NULL UNIQUE,
    abstract TEXT,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

db.run("""
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

db.run("""
CREATE TABLE IF NOT EXISTS paper_projects (
    paper_id INTEGER NOT NULL,
    project_id INTEGER NOT NULL,
    PRIMARY KEY (paper_id, project_id),
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
""")

db.run("""
INSERT OR IGNORE INTO projects (name) VALUES
('LLM Safety'),
('MLLM Long Context'),
('AI Hallucination'),
('Medical AI');
""")


# Tool: Add paper directly if web search off
@tool
def add_paper_direct(title: str, link: str = "") -> str:
    """
    Add a Paper title to the database.

    Args: 
        title: Title of the research paper.
    """

    safe_title = (title or "").replace("'", "''").strip()
    safe_link = (link or "").replace("'", "''").strip()

    try:
        db.run(f"""
        INSERT OR IGNORE INTO papers (title, link)
        VALUES ('{safe_title}', '{safe_link}');
        """)
        return f"Successfully added the paper '{title}'"
    except Exception as e:
        return f"Failed to add the title: {str(e)}"
    
# Tool: Search Paper online
@tool
def find_paper_online(title:str) -> str:
    """
    Search the web for a research paper by title and store possible matches for confirmation.
    
    Args: 
        title: Title of the research paper
    """

    #ei jinish search korbe ai
    query= f"{title} research paper arxiv"

    try:
        results = search.results(query)
    except Exception as e:
        return f"Web Search failed {str(e)}"
    
    # ekhane "organic" ekta peram jeta bujhay normal search results gula jeno dekhay not ads
    # ekhane organic or normal results na paile empty string nibe
    # get() user korle always empty value handle korte hobe otherwise crash korbe
    organic_results = results.get("organic", [])

    if not organic_results:
        return f"No web search found for the {title}"
    
    # proti title er shathe jodi link pay tailei ei list e rakhbo, naile na
    matches = []
    for i in organic_results[:5]:
        candidate_title = i.get("title", "").strip()
        candidate_link = i.get("link", "").strip()

        # proti title er shathe jodi link pay tailei ei list e rakhbo, naile na
        if candidate_title and candidate_link:
            matches.append({
                "title": candidate_title,
                "link": candidate_link
            })

    if not matches:
        return f"No useful matches found for '{title}'."
        
    

    st.session_state.pending_paper_matches = matches

    response =  [f"I have found possible papers for {title}"]

    # enumerate ittem and number duitai pathay.
    #ekhane item 1 theke shuru
    for i, match in enumerate(matches, start = 1):
        response.append(f"{i}. {match['title']}\n{match['link']}")

    response.append("Reply with something like: confirm paper 1")
    return "\n\n".join(response)


# Confirm korar por add kora
@tool
def confirm_add_paper(choice: int) -> str:
    """
    Confirm one of the previously found paper matches and add it to the database.
    Args:
        choice: Number of the paper to add from the suggested list
    """
    
    matches = st.session_state.pending_paper_matches

    if choice < 1 or choice > 5:
        return f"Invalid paper number and insert a value between 1 to 5"
    else:
        title = matches[choice-1]["title"]
        link = matches[choice-1]["link"]
    
    try:
        db.run(f"""
            INSERT OR IGNORE INTO papers (title, link)
            VALUES ('{title}', '{link}');
        """)
        st.session_state.pending_paper_matches = []
        return f"Paper '{title}' added successfully."
    except Exception as e:
        return f"Failed to add paper '{title}'. Error: {str(e)}"   


llm = ChatGroq(model="openai/gpt-oss-20b")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = toolkit.get_tools()
custom_tools = [add_paper_direct]
search = GoogleSerperAPIWrapper()

system_prompt = """
You are a research paper tracker assistant.

Rules:
1. If web search is enabled and the user asks to add a paper, first use the find_paper_online tool.
2. If exact or similar matches are found, ask the user to confirm one option.
3. If the user says something like 'confirm paper 1', use the confirm_add_paper tool.
4. If web search is disabled, use add_paper_direct and save the title exactly as given.
5. Be concise and clear.
"""

def build_agent(enable_web_search: bool):
    if enable_web_search:
        agent_tools = [find_paper_online, confirm_add_paper, add_paper_direct]
    else:
        agent_tools = [add_paper_direct]

    agent = create_agent(
        model=llm,
        tools=agent_tools,
        checkpointer=st.session_state.memory,
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
        {"configurable": {"thread_id": "rafin420"}}
    )

    st.chat_message("ai").markdown(res["messages"][-1].content)
    st.session_state.messages.append({"role": "ai", "content": res["messages"][-1].content})
        
    

