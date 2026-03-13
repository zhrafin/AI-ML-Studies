# Research Paper Tracker AI

*A simple Streamlit chat app for managing research papers*

---

## Overview

I am building a **Streamlit chat application** that helps me track research papers and organize them by research projects.

The system works like a small personal research assistant. I can add papers, store summaries, assign them to projects, and search for new papers from the web.

The design stays simple at the beginning. I focus on reliable paper tracking first. I add semantic similarity later as a second phase.

---

## Development Strategy

I split the project into two phases.

### Phase 1, Core Tracker

I first build a working research tracker with basic functionality.

Features include:

* Paper storage
* CRUD operations
* Project organization
* Web search for new papers
* Abstract fetching and summarization
* Manual saving of selected papers

No semantic similarity in this phase.

### Phase 2, Semantic Similarity

After the base system works, I add semantic search.

Approach:

* Combine paper text fields
  `title + abstract + summary`
* Generate embeddings
* Store embeddings in a vector index
* Search the vector index for similar papers
* Retrieve full records from SQLite

This keeps the system simple and avoids forcing semantic search inside SQLite.

---

## Main Goal

I want a **single-file style Streamlit chat app** that behaves like a lightweight LangChain agent.

The application helps me:

* track research papers
* organize papers by research projects
* search the web for related papers
* summarize abstracts
* store selected results

---

## Paper Data Model

Each paper stores the following fields:

* **title**
* **abstract**
* **summary**

The title must be unique to prevent duplicate papers.

---

## Project System

I organize papers using research projects.

### Projects Table

Each project represents a research direction or topic.

Examples:

* Retrieval Augmented Generation
* LLM Alignment
* Multimodal Models

### Many-to-Many Relationship

A single paper may belong to multiple projects.

Example:

A paper about multimodal reasoning might belong to:

* Multimodal Models
* LLM Reasoning

To support this, I use a linking table between papers and projects.

---

## Main Application Capabilities

The application allows me to perform the following actions.

### Paper Management

* Create paper
* Read paper list
* Update paper
* Delete paper

### Project Management

* Create projects manually
* Assign papers to projects
* List papers by project

### Duplicate Prevention

The system checks paper titles before insertion to prevent duplicates.

### Abstract Processing

The system can:

* fetch abstract from web
* summarize the abstract

### Web Search for Papers

The agent can search the web for related papers.

Workflow:

1. Search using Google Serper
2. Return top results
3. Show numbered list

Example:

```
1. Paper title
2. Paper title
3. Paper title
4. Paper title
5. Paper title
```

I can then save selected papers using:

```
save 1, 3, 4
```

Only the selected results are stored.

---

## Project Creation Rule

Projects are usually created manually.

If a paper references a project that does not exist, the model must ask before creating it.

Example:

```
Project "Multimodal Reasoning" does not exist.
Do you want me to create it?
```

---

## Technology Stack

The application uses simple and reliable tools.

### Storage

* **SQLite**
* structured relational tables

### Interface

* **Streamlit chat UI**

### Web Search

* **Google Serper API**

### Agent Logic

* LangChain style tool-based agent

### Future Semantic Search

* embeddings
* lightweight vector index

---

## Database Schema

I start with these tables.

### Papers Table

Stores research paper information.

Fields:

* id
* title
* abstract
* summary

### Projects Table

Stores research projects.

Fields:

* id
* project_name

### Paper Projects Table

Many-to-many mapping between papers and projects.

Fields:

* paper_id
* project_id

### Optional Future Table

For semantic similarity.

`paper_embeddings`

Fields:

* paper_id
* embedding_vector

---

## Application Structure

I keep the project simple and modular.

### 1. Imports

Main dependencies:

* dotenv
* streamlit
* sqlite
* LangChain components
* Google Serper
* embedding utilities later

---

### 2. Streamlit Setup

The UI initializes:

* application title
* web search enable checkbox
* session state
* chat history rendering

---

### 3. Database Setup

The app creates required tables if they do not exist.

Tables include:

* `papers`
* `projects`
* `paper_projects`

Future table:

* `paper_embeddings`

---

### 4. Helper Functions

Core database operations live here.

Examples:

* create_paper()
* update_paper()
* delete_paper()
* get_papers()
* get_papers_by_project()
* check_duplicate_title()
* create_project()
* assign_paper_to_project()
* fetch_web_papers()
* save_selected_papers()

These functions stay simple and readable.

---

### 5. Tools

Helper functions are wrapped as **agent tools**.

Main tool groups:

**Database Tools**

* paper CRUD
* project management
* paper assignment

**Web Tools**

* search papers from web
* fetch abstract
* summarize abstract

**Future Tool**

* semantic similarity search

---

### 6. System Prompt

The system prompt defines rules for the agent.

Important rules include:

* prevent duplicate titles
* ask before creating missing projects
* show numbered web search results
* never auto-save web search results
* only save papers when user selects them

---

### 7. Agent Builder

The application creates an agent dynamically.

Example structure:

```
build_agent(enable_web_search: bool)
```

This allows web search to be toggled from the UI.

---

### 8. Chat Flow

The chat loop works like this.

1. User enters message
2. Agent processes request
3. Agent calls tools if needed
4. Final response is returned
5. Response is displayed
6. Message history updates in session state

---

## Development Rule

To keep the project manageable, I follow this order:

1. Database schema
2. Helper functions
3. Tools
4. Agent
5. Streamlit UI

This prevents confusion and keeps the system stable.

---

## Small Code Fix

One correction in the current code pattern.

Incorrect import:

```python
from streamlit import st
```

Correct import:

```python
import streamlit as st
```

Another issue:

```python
tools = [search.run, tools]
```

This creates a nested list.

The correct approach is to combine tool lists properly.

---

## Summary

This project is a **personal research paper tracker with agent capabilities**.

Key design principles:

* start simple
* build reliable CRUD system first
* add semantic similarity later
* keep the code modular
* maintain a clear database structure

The result will be a lightweight tool that helps me manage and explore research papers efficiently.
