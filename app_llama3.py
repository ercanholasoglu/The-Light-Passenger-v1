import streamlit as st
import time
import getpass
from neo4j import GraphDatabase
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages as lg_add_messages
import sys
import re
import requests
from cachetools import cached, TTLCache
import graphviz
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from operator import add
import unicodedata
from langgraph.checkpoint.memory import MemorySaver
import random
import os
import uuid
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import subprocess

# Ortam değişkenlerini al
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "llama3-8b-tr")

# ---------------- Utility functions ----------------
def sanitize_markdown(text):
    if not isinstance(text, str):
        return str(text)
    if not text:
        return ""
    # Escape HTML-sensitive characters and some markdown characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    markdown_chars = ['\\', '*', '_', '~', '`', '#', '[', ']', '(', ')', '{', '}', '!', '^']
    for char in markdown_chars:
        text = text.replace(char, f"\\{char}")
    return text

def safe_markdown(text):
    # placeholder - keep as-is for now
    return text

def get_ollama_client():
    # Önce environment variable kontrol et
    model_name = os.getenv("OLLAMA_MODEL_NAME")
    if model_name:
        return model_name

    # Eğer set edilmemişse ollama list komutunu çalıştır
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()

        # Hugging Face'den çektiğin modeli ara
        for line in output.splitlines():
            if "fine_tuned_llama3_8b_for_tr_gguf" in line:
                model_name = line.split()[0]
                return model_name

        # Eğer hiç bulamazsa fallback
        return "llama3"
    except Exception as e:
        print(f"Ollama list okunamadı: {e}")
        return "llama3"


class Neo4jConnector:
    def __init__(self):
        self.uri = NEO4J_URI
        self.user = NEO4J_USER
        self.password = NEO4J_PASSWORD
        self.database = "neo4j"
        self.driver = None

    def connect(self):
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                # quick connectivity check
                with self.driver.session(database=self.database) as session:
                    session.run("RETURN 1")
            except Exception as exc:
                raise ConnectionError(f"Neo4j bağlantı hatası: {exc}") from exc

    def save_chat_message(self, session_id: str, role: str, content: str, timestamp: datetime):
        # This method may raise; callers should wrap it in try/except if they don't want app to crash.
        self.connect()
        query = """
        MERGE (s:ChatSession {id: $session_id})
        CREATE (m:ChatMessage {
            role: $role,
            content: $content,
            timestamp: $timestamp
        })
        MERGE (s)-[:HAS_MESSAGE]->(m)
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, session_id=session_id, role=role, content=content, timestamp=timestamp.isoformat())

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

# Safe wrapper so DB issues don't break UI
def safe_save_chat(neo4j_connector, session_id, role, content, timestamp):
    try:
        if neo4j_connector is None:
            return
        neo4j_connector.save_chat_message(session_id, role, content, timestamp)
    except Exception as e:
        # Print to server logs; do not interrupt the app
        print(f"[Warning] Neo4j kaydı başarısız: {e}")

# ---------------- Data fetch helper (unchanged) ----------------
DATABASE = "moviesandseries"

query = """
MATCH (m:Movie)
OPTIONAL MATCH (m)-[:HAS_LANGUAGE]->(l:Language)
OPTIONAL MATCH (m)-[:IN_GENRE]->(g:Genre)
OPTIONAL MATCH (m)-[:MADE_IN]->(co:Country)
OPTIONAL MATCH (m)<-[:PRODUCED]-(comp:Company)
OPTIONAL MATCH (m)-[rel_avail:AVAILABLE_ON]->(p:Platform)
OPTIONAL MATCH (m)<-[rel_acted:ACTED_IN]-(a:Person)
OPTIONAL MATCH (m)<-[rel_worked:WORKED_AS]-(cr:Person)
RETURN 
    m { 
        .title, 
        .imdb_id,
        .imdb_rating, 
        .imdb_votes, 
        .metacritic,
        .box_office,
        .awards,
        .rotten_tomatoes,
        .homepage,
        .budget,
        .revenue,
        .vote_average,
        .vote_count,
        .runtime,
        .poster,
        .rated,
        .overview
    } AS movie_props, 
    collect(DISTINCT {name: l.name, code: l.code}) AS languages, 
    collect(DISTINCT g.name) AS genres, 
    collect(DISTINCT co.name) AS countries, 
    collect(DISTINCT comp.name) AS companies, 
    collect(DISTINCT {
        name: a.name, 
        character: rel_acted.character, 
        order: rel_acted.order
    }) AS actors,
    collect(DISTINCT {
        name: cr.name, 
        job: rel_worked.job, 
        department: rel_worked.department
    }) AS crew,
    collect(DISTINCT {
        platform_name: p.name, 
        country: rel_avail.country, 
        type: rel_avail.type
    }) AS platforms
"""

def fetch_all_movie_data_with_details(uri, username, password, db_name, cypher_query):
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        print(f"✅ Neo4j '{db_name}' veritabanına başarıyla bağlandı!")
        all_data = []
        with driver.session(database=db_name) as session:
            result = session.run(cypher_query)
            for record in result:
                movie_data = dict(record["movie_props"])
                movie_data["genres"] = record["genres"]
                movie_data["languages"] = record["languages"]
                movie_data["countries"] = record["countries"]
                movie_data["companies"] = record["companies"]
                movie_data["actors"] = record["actors"]
                movie_data["crew"] = record["crew"]
                movie_data["platforms"] = record["platforms"]
                all_data.append(movie_data)
        return all_data
    except Exception as e:
        print(f"❌ Veri çekme hatası: {e}")
        # Return empty list instead of exiting to avoid killing the app during startup
        return []
    finally:
        if driver:
            driver.close()
            print("✅ Bağlantı kapatıldı.")

movies_data = fetch_all_movie_data_with_details(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATABASE, query)

# ---------------- LangGraph / Agent helpers ----------------
def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    return left + right

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    last_recommended_place: Optional[str]

# ---------------- LLM helpers: cached client, timeouted invoke, safe_stream ----------------
_executor = ThreadPoolExecutor(max_workers=2)

@st.cache_resource
def get_ollama_client():
    model_name = OLLAMA_MODEL_NAME
    return ChatOllama(model=model_name, temperature=0)

def invoke_llm_with_timeout(messages, timeout_seconds=30):
    client = get_ollama_client()
    # submit the client.invoke call -- many client.invoke implementations accept messages in a list/dict form
    future = _executor.submit(client.invoke, messages)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeout:
        future.cancel()
        raise TimeoutError(f"LLM çağrısı {timeout_seconds}s içinde tamamlanmadı.")
    except Exception:
        # pass through other exceptions to caller
        raise

def safe_stream(app, payload, config=None, overall_timeout=60):
    """
    Run app.stream in background thread and yield items, raising TimeoutError on overall timeout.
    """
    q = queue.Queue()

    def runner():
        try:
            for s in app.stream(payload, config=config):
                q.put(("item", s))
            q.put(("done", None))
        except Exception as e:
            q.put(("error", e))

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    start = time.time()
    while True:
        try:
            kind, val = q.get(timeout=1)
            if kind == "item":
                yield val
            elif kind == "done":
                return
            elif kind == "error":
                raise val
        except queue.Empty:
            if time.time() - start > overall_timeout:
                raise TimeoutError("Stream zaman aşımına uğradı.")
            continue

# ---------------- Core agent logic ----------------
def generate_response(state: AgentState) -> AgentState:
    messages = state['messages']
    try:
        response = invoke_llm_with_timeout(messages, timeout_seconds=30)
        # response might be an object with .content or dict-like
        content = None
        if hasattr(response, "content"):
            content = response.content
        else:
            try:
                content = response.get("content") if isinstance(response, dict) else str(response)
            except Exception:
                content = str(response)
        return {"messages": [AIMessage(content=content)]}
    except TimeoutError as te:
        return {"messages": [AIMessage(content=f"Üzgünüm, yanıt zaman aşımına uğradı: {te}")] }
    except Exception as e:
        return {"messages": [AIMessage(content=f"Üzgünüm, şu an bir yanıt oluşturamıyorum. Hata: {e}")]}

def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_response", generate_response)
    workflow.set_entry_point("generate_response")
    workflow.add_edge("generate_response", END)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="The Light Passenger", layout="wide")
st.title("The Light Passenger 📍")

if "messages" not in st.session_state:
    st.session_state.messages = []

# normalize stored messages (if they were saved as dicts)
for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, dict):
        if msg.get("role") == "user":
            st.session_state.messages[i] = HumanMessage(content=msg.get("content", ""))
        elif msg.get("role") == "assistant":
            st.session_state.messages[i] = AIMessage(content=msg.get("content", ""))

neo4j_connector = Neo4jConnector()

# render existing messages
for message in st.session_state.messages:
    display_role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(display_role):
        st.markdown(message.content, unsafe_allow_html=True)

# chat input
if prompt := st.chat_input("Nasıl yardımcı olabilirim?"):
    if "conversation_thread_id" not in st.session_state:
        st.session_state.conversation_thread_id = str(uuid.uuid4())
    session_id = st.session_state.conversation_thread_id
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    # use safe save so DB problems don't crash UI
    safe_save_chat(neo4j_connector, session_id, "user", prompt, datetime.now())
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Yanıt oluşturuluyor..."):
        app = create_workflow()
        config = {"configurable": {"thread_id": session_id}}
        latest_ai_message_content = ""
        try:
            # use safe_stream to avoid blocking indefinitely
            for s in safe_stream(app, {"messages": st.session_state.messages}, config=config, overall_timeout=60):
                for key in s:
                    node_output = s[key]
                    if "messages" in node_output:
                        for msg in reversed(node_output["messages"]):
                            if isinstance(msg, AIMessage) and getattr(msg, "content", None):
                                latest_ai_message_content = msg.content
                                break
            if latest_ai_message_content:
                sanitized_content = sanitize_markdown(latest_ai_message_content)
                ai_message = AIMessage(content=sanitized_content)
                st.session_state.messages.append(ai_message)
                safe_save_chat(neo4j_connector, session_id, "assistant", sanitized_content, datetime.now())
                with st.chat_message("assistant"):
                    st.markdown(sanitized_content, unsafe_allow_html=True)
            else:
                error_msg = "Üzgünüm, bir yanıt üretemedim. LangGraph akışı tamamlandı ancak yapay zeka mesajı bulunamadı. Lütfen tekrar deneyin."
                ai_error_message = AIMessage(content=error_msg)
                st.session_state.messages.append(ai_error_message)
                safe_save_chat(neo4j_connector, session_id, "assistant", error_msg, datetime.now())
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.error("LangGraph akışı yapay zeka mesajı üretmeden tamamlandı.")
        except Exception as e:
            error_message = f"Bir hata oluştu: {e}. Lütfen daha sonra tekrar deneyin."
            st.error(f"Ana döngüde beklenmedik hata: {str(e)}")
            print(f"ERROR in main loop: {str(e)}")
            ai_error_message = AIMessage(content=error_message)
            st.session_state.messages.append(ai_error_message)
            safe_save_chat(neo4j_connector, session_id, "assistant", error_message, datetime.now())
            with st.chat_message("assistant"):
                st.markdown(error_message)
                st.exception(e)

# removed unconditional st.rerun() to avoid infinite reload loop
