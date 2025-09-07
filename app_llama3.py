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
from langgraph.graph.message import add_messages
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

# Ortam değişkenlerini al
try:
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USER = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    OLLAMA_MODEL_NAME = "llama3-8b-tr"
except KeyError as e:
    st.error(f"Eksik ortam değişkeni: {e}. Lütfen yapılandırın.")
    st.stop()


def sanitize_markdown(text):
    if not isinstance(text, str):
        return str(text)
    if not text:
        return ""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    markdown_chars = ['\\', '*', '_', '~', '`', '#', '[', ']', '(', ')', '{', '}', '!', '^']
    for char in markdown_chars:
        text = text.replace(char, f"\\{char}")
    return text

def safe_markdown(text):
    return text

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
                with self.driver.session(database=self.database) as session:
                    session.run("RETURN 1") 
            except Exception as exc:
                raise ConnectionError(f"Neo4j bağlantı hatası: {exc}") from exc
            
    def save_chat_message(self, session_id: str, role: str, content: str, timestamp: datetime):
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
        try:
            with self.driver.session(database=self.database) as session:
                session.run(query, session_id=session_id, role=role, content=content, timestamp=timestamp.isoformat())
        except Exception as exc:
            st.error(f"Neo4j mesaj kaydetme hatası: {exc}")
            print(f"Neo4j mesaj kaydetme hatası: {exc}")
            
    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None
            
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
        sys.exit()
    finally:
        if driver:
            driver.close()
            print("✅ Bağlantı kapatıldı.")

movies_data = fetch_all_movie_data_with_details(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATABASE, query)

def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    return left + right

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    last_recommended_place: Optional[str]

def generate_response(state: AgentState) -> AgentState:
    messages = state['messages']
    
    llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0)

    try:
        response = llm.invoke(messages)
        return {"messages": [AIMessage(content=response.content)]}
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

st.set_page_config(page_title="The Light Passenger", layout="wide")
st.title("The Light Passenger 📍")

if "messages" not in st.session_state:
    st.session_state.messages = []
for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, dict):
        if msg["role"] == "user":
            st.session_state.messages[i] = HumanMessage(content=msg["content"])
        elif msg["role"] == "assistant":
            st.session_state.messages[i] = AIMessage(content=msg["content"])
neo4j_connector = Neo4jConnector()

for message in st.session_state.messages:
    display_role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(display_role):
        st.markdown(message.content, unsafe_allow_html=True)

if prompt := st.chat_input("Nasıl yardımcı olabilirim?"):
    if "conversation_thread_id" not in st.session_state:
        st.session_state.conversation_thread_id = str(uuid.uuid4())
    session_id = st.session_state.conversation_thread_id
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    neo4j_connector.save_chat_message(session_id, "user", prompt, datetime.now())
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Yanıt oluşturuluyor..."):
        app = create_workflow()
        config = {"configurable": {"thread_id": session_id}}
        latest_ai_message_content = ""
        try:
            for s in app.stream({"messages": st.session_state.messages}, config=config):
                for key in s:
                    node_output = s[key]
                    if "messages" in node_output:
                        for msg in reversed(node_output["messages"]):
                            if isinstance(msg, AIMessage) and msg.content:
                                latest_ai_message_content = msg.content
                                break
            if latest_ai_message_content:
                sanitized_content = sanitize_markdown(latest_ai_message_content)
                ai_message = AIMessage(content=sanitized_content)
                st.session_state.messages.append(ai_message)
                neo4j_connector.save_chat_message(session_id, "assistant", sanitized_content, datetime.now())
                with st.chat_message("assistant"):
                    st.markdown(sanitized_content, unsafe_allow_html=True)
            else:
                error_msg = "Üzgünüm, bir yanıt üretemedim. LangGraph akışı tamamlandı ancak yapay zeka mesajı bulunamadı. Lütfen tekrar deneyin."
                ai_error_message = AIMessage(content=error_msg)
                st.session_state.messages.append(ai_error_message)
                neo4j_connector.save_chat_message(session_id, "assistant", error_msg, datetime.now())
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.error("LangGraph akışı yapay zeka mesajı üretmeden tamamlandı.")
        except Exception as e:
            error_message = f"Bir hata oluştu: {e}. Lütfen daha sonra tekrar deneyin."
            st.error(f"Ana döngüde beklenmedik hata: {str(e)}")
            print(f"ERROR in main loop: {str(e)}")
            ai_error_message = AIMessage(content=error_message)
            st.session_state.messages.append(ai_error_message)
            neo4j_connector.save_chat_message(session_id, "assistant", error_message, datetime.now())
            with st.chat_message("assistant"):
                st.markdown(error_message)
                st.exception(e)
    st.rerun()
