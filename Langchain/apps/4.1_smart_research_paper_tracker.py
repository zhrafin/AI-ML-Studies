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
    
@tool
def list_papers():
    """
    List all the saved papers
    """
    try:
        results = db.run("""
        SELECT title, link
        FROM papers
        ORDER BY created_at DESC;
        """)
        return results
    except Exception as e:
        return f"Failed to list {str(e)}"
    

@tool 
def assign_paper_to_project(title:str, project_name:str):
    """
    Assign a saved paper to a project.
    Args:
        title: Title of the research paper
        project_name: Name of the project 
    """
    try:
        paper_result = db.run(f"""
        SELECT id FROM papers
        WHERE title = '{title}'
        LIMIT 1;
        """)
        project_result = db.run(f"""
        SELECT id FROM projects
        WHERE name = '{project_name}'
        LIMIT 1;
        """)

        paper_id = paper_result.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "")
        project_id = project_result.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "")
        
        db.run(f"""
        INSERT OR IGNORE INTO paper_projects (paper_id, project_id)
        VALUES ({paper_id}, {project_id});
        """)
        
        return f"Paper '{title}' has been assigned to project '{project_name}'."

    except Exception as e:
        return f"Failed to assign paper to project. Error: {str(e)}"

@tool
def summarize_abstract(title: str, abstract: str):
    """
    Summarize the research paper abstract
    Args:
        title: The title of the research paper
        abstract: This is the abstract of the research paper
    """
    try:
        response=llm.invoke("""
            Summarize the following research paper abstract in 3 to 5 simple sentences.
            Focus on:
            1. What the paper is about
            2. What method or idea it uses
            3. Why it matters
        """)
        return response.content
    except Exception as e:
        return f"failed to summarize the abstract. Error {str(e)}"


@tool
def add_summary_abstract_to_db(title: str, abstract: str):
    """
    It will add the summarized abstract to the database
    Args:
        title: The title of the research paper
        abstract: This is the summarized abstract of the research paper
    """

    try:
        db.run(f"""
        UPDATE papers
        SET abstract = '{abstract}',
            updated_at = CURRENT_TIMESTAMP
        WHERE title = '{title}';
        """         
        )
        return "Abstract Summary added successfully"
    except Exception as e:
        return f"Error adding Abstract {str(e)}"


llm = ChatGroq(model="openai/gpt-oss-20b")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
search = GoogleSerperAPIWrapper()

system_prompt = """
You are a research paper tracker assistant.

Rules:
1. If web search is enabled and the user asks to add a paper, use find_paper_online first.
2. If matches are found, ask the user to confirm one option.
3. If the user replies with a number like 1 or 2, use confirm_add_paper with that number.
4. If web search is disabled, use add_paper_direct and save the title exactly as given.
5. If the user asks to show, list, or display saved papers, use list_papers.
6. If the user asks to assign a paper to a project, use assign_paper_to_project.
7. If the user provides an abstract and asks for a summary, use summarize_abstract.
8. Do not invent links.
9. Be concise and clear.
"""

def build_agent(enable_web_search: bool):
    if enable_web_search:
        agent_tools = [find_paper_online, 
                       confirm_add_paper, 
                       add_paper_direct, 
                       list_papers, 
                       assign_paper_to_project, 
                       summarize_abstract]
    else:
        agent_tools = [add_paper_direct, 
                       list_papers, 
                       assign_paper_to_project, 
                       summarize_abstract]

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
        
    

