import streamlit as st
import time
import getpass
from neo4j import GraphDatabase
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

# ---------------- Ortam değişkenleri ----------------
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
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    markdown_chars = ['\\', '*', '_', '~', '`', '#', '[', ']', '(', ')', '{', '}', '!', '^']
    for char in markdown_chars:
        text = text.replace(char, f"\\{char}")
    return text

def safe_markdown(text):
    return text

def get_ollama_client():
    return ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.7)

# ---------------- Neo4j Connector ----------------
class Neo4jConnector:
    def __init__(self, db_name="neo4j"):
        self.uri = NEO4J_URI
        self.user = NEO4J_USER
        self.password = NEO4J_PASSWORD
        self.database = db_name
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
        with self.driver.session(database=self.database) as session:
            session.run(query, session_id=session_id, role=role, content=content, timestamp=timestamp.isoformat())

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

def safe_save_chat(neo4j_connector, session_id, role, content, timestamp):
    try:
        if neo4j_connector:
            neo4j_connector.save_chat_message(session_id, role, content, timestamp)
    except Exception as e:
        print(f"[Warning] Neo4j kaydı başarısız: {e}")

# ---------------- Data fetch ----------------
DATABASE = "moviesandseries"

movie_query = """
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

places_query = """
MATCH (p:Place)
RETURN 
    p.name AS name,
    p.google_adres AS google_adres,
    p.fiyat_seviyesi_simge AS fiyat_seviyesi_simge,
    p.google_ortalama_puan AS google_ortalama_puan,
    p.google_telefon AS google_telefon,
    p.google_toplam_yorum AS google_toplam_yorum,
    p.google_web_sitesi AS google_web_sitesi,
    p.maps_linki AS maps_linki,
    p.google_fotograf_linkleri AS google_fotograf_linkleri,
    p.original_id AS original_id
"""

def fetch_all_movie_data_with_details(uri, username, password, db_name, cypher_query):
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
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
        return []
    finally:
        if driver:
            driver.close()

def fetch_all_place_data(uri, username, password, db_name, cypher_query):
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        all_data = []
        with driver.session(database=db_name) as session:
            result = session.run(cypher_query)
            for record in result:
                all_data.append(dict(record))
        return all_data
    except Exception as e:
        print(f"❌ Mekan verisi çekme hatası: {e}")
        return []
    finally:
        if driver:
            driver.close()

movies_data = fetch_all_movie_data_with_details(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATABASE, movie_query)
places_data = fetch_all_place_data(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, "neo4j", places_query)

# ---------------- LangGraph / Agent helpers ----------------
def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    return left + right

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    last_recommended_place: Optional[str]

_executor = ThreadPoolExecutor(max_workers=2)

# ---------------- Core agent logic ----------------
system_message_template = """
Sen bir sohbet robotusun ve kullanıcıya film, dizi veya İstanbul'da mekan önerileri sunuyorsun.
Tüm yanıtların Türkçe olacak.
Eğer kullanıcı İstanbul'da bir yer veya mekan sorarsa:
- Yalnızca Neo4j'deki `places_data` verisini kullan,
- Mekanın adı, adresi, fiyat seviyesi, puanı, yorum sayısı, telefon, web sitesi ve maps linkini özetle,
- Gereksiz hayali mekan ekleme.
Eğer kullanıcı film veya dizi ile ilgili bir soru sorarsa:
- Yalnızca `movies_data` verisini kullan.
Eğer bilgi yoksa dürüstçe söyle.
"""

def generate_response(state: AgentState) -> AgentState:
    agent_state = st.session_state.agent_state
    user_message = state['messages'][-1].content
    agent_state['messages'].append(HumanMessage(content=user_message))
    
    last_user_message_lower = user_message.lower() if user_message else ""
    selected_places = []

    is_place_request = any(k in last_user_message_lower for k in ["mekan", "yer", "restoran", "bar", "cafe", "randevu", "date"])

    if is_place_request and places_data:
        semt_keywords = ["suadiye", "kadıköy", "beşiktaş", "beyoğlu", "şişli", "ümraniye", "eyüpsultan", "göktürk"]
        user_semt = next((k for k in semt_keywords if k in last_user_message_lower), None)

        if user_semt:
            selected_places = [p for p in places_data 
                               if p.get('google_adres') and re.search(user_semt, p.get('google_adres','').lower())]
        else:
            selected_places = sorted([p for p in places_data if p.get('google_ortalama_puan')], 
                                     key=lambda x: x.get('google_ortalama_puan',0), reverse=True)[:5]

        if selected_places:
            place_info = []
            for p in selected_places:
                fotos = p.get('google_fotograf_linkleri', [])
                if fotos and all(isinstance(c,str) and len(c)==1 for c in fotos):
                    fotos = ["".join(fotos)]
                fotos_text = ", ".join(fotos[:3]) if fotos else "-"
                place_info.append(
                    f"{p.get('name','-')}\n"
                    f"Adres: {p.get('google_adres','-')}\n"
                    f"Fiyat Seviyesi: {p.get('fiyat_seviyesi_simge','-')}\n"
                    f"Puan: {p.get('google_ortalama_puan','-')}\n"
                    f"Yorum Sayısı: {p.get('google_toplam_yorum','-')}\n"
                    f"Telefon: {p.get('google_telefon','-')}\n"
                    f"Web Sitesi: {p.get('google_web_sitesi','-')}\n"
                    f"Harita Linki: {p.get('maps_linki','-')}\n"
                    f"Fotoğraflar: {fotos_text}\n"
                )
            prompt_with_data = f"{system_message_template}\n\nİstanbul'daki uygun mekanlar:\n" + "\n".join(place_info) + f"\nKullanıcı: {user_message}"
        else:
            prompt_with_data = f"{system_message_template}\n\nÜzgünüm, '{user_semt}' semtinde uygun bir mekan bulunamadı.\nKullanıcı: {user_message}"

    else:
        movie_info = "\n".join([
            f"{m.get('title','-')}, Türler: {', '.join(m.get('genres',[]))}, "
            f"IMDb: {m.get('imdb_rating','-')}"
            for m in movies_data
        ])
        prompt_with_data = f"{system_message_template}\n\nFilm ve dizi verileri:\n{movie_info}\nKullanıcı: {user_message}"

    try:
        response = st.session_state.ollama_client.invoke([HumanMessage(content=prompt_with_data)], temperature=0.7)
        content = response.content.strip() or "Üzgünüm, şu anda yanıt oluşturamıyorum."
        agent_state['messages'].append(AIMessage(content=content))
        return agent_state
    except Exception as e:
        agent_state['messages'].append(AIMessage(content=f"Üzgünüm, yanıt oluşturulamadı: {e}"))
        return agent_state

# ---------------- Streamlit App ----------------
st.title("İstanbul Film & Mekan Chatbotu")

if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {'messages': [], 'last_recommended_place': None}
if 'ollama_client' not in st.session_state:
    st.session_state.ollama_client = get_ollama_client()

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Sorunuzu yazın ve Enter'a basın:")
    submitted = st.form_submit_button("Gönder")

if submitted and user_input:
    state = {'messages': [HumanMessage(content=user_input)]}
    updated_state = generate_response(state)

    for msg in updated_state['messages'][-2:]:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**Siz:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Bot:** {msg.content}")
