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
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    markdown_chars = ['\\', '*', '_', '~', '`', '#', '[', ']', '(', ')', '{', '}', '!', '^']
    for char in markdown_chars:
        text = text.replace(char, f"\\{char}")
    return text

def safe_markdown(text):
    return text

def get_ollama_client():
    model_name = OLLAMA_MODEL_NAME
    return ChatOllama(model=model_name, temperature=0)

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
    p.google_adres AS adres,
    p.fiyat_seviyesi_simge AS fiyat,
    p.google_ortalama_puan AS puan,
    p.google_telefon AS telefon,
    p.google_toplam_yorum AS yorum_sayisi,
    p.google_web_sitesi AS web,
    p.maps_linki AS maps_linki,
    p.google_fotograf_linkleri AS fotograflar,
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

def invoke_llm_with_timeout(messages, timeout_seconds=120):
    client = ollama_client  # artık st.session_state kullanmıyoruz
    future = _executor.submit(client.invoke, messages)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeout:
        future.cancel()
        raise TimeoutError(f"LLM çağrısı {timeout_seconds}s içinde tamamlanmadı.")
    except Exception:
        raise


def safe_stream(app, payload, config=None, overall_timeout=60):
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
system_message_template = """
Sen bir sohbet robotusun ve kullanıcıya film, dizi veya İstanbul'da mekan önerileri sunuyorsun.
Tüm yanıtların Türkçe olacak.
Eğer kullanıcı İstanbul'da bir yer veya mekan sorarsa:
- Yalnızca Neo4j'deki `places_data` verisini kullan,
- Kullanıcının belirttiği semte uygun mekanları filtrele,
- Mekanın adı, adresi, fiyat seviyesi, puanı, yorum sayısı, telefon, web sitesi ve maps linkini özetle,
- Gereksiz hayali mekan ekleme.
Eğer kullanıcı film veya dizi ile ilgili bir soru sorarsa:
- Yalnızca `movies_data` verisini kullan.
Eğer bilgi yoksa dürüstçe söyle.
"""
def generate_response(state: AgentState) -> AgentState:
    messages = state['messages']
    last_user_message = messages[-1].content.lower()
    selected_places = []

    is_place_request = any(k in last_user_message for k in ["mekan", "yer", "restoran", "bar", "cafe", "randevu", "date"])

    if is_place_request and places_data:
        semt_keywords = ["suadiye", "kadıköy", "beşiktaş", "beyoğlu", "şişli", "ümraniye", "eyüpsultan", "göktürk"]
        user_semt = next((k for k in semt_keywords if k in last_user_message), None)

        if user_semt:
            selected_places = [p for p in places_data if user_semt in p.get('google_adres','').lower()]
        else:
            # Semt yoksa popüler mekanları puana göre sırala ve top 5 ver
            selected_places = sorted(places_data, key=lambda x: x.get('google_ortalama_puan',0), reverse=True)[:5]

        if not selected_places:
            prompt_with_data = f"{system_message_template}\n\nÜzgünüm, '{user_semt}' semtinde uygun bir mekan bulunamadı. İstanbul genelinden öneri ister misiniz?\nKullanıcı: {last_user_message}"
        else:
            place_info = []
            for p in selected_places:
                fotos = p.get('google_fotograf_linkleri', [])
                # Eğer liste karakter bazlı geliyorsa join et
                if fotos and all(isinstance(c,str) and len(c)==1 for c in fotos):
                    fotos = ["".join(fotos)]
                fotos_text = ", ".join(fotos[:3]) if fotos else "-"
                place_info.append(
                    f"• {p.get('name','-')}\n"
                    f"  Adres: {p.get('google_adres','-')}\n"
                    f"  Fiyat Seviyesi: {p.get('fiyat_seviyesi_simge','-')}\n"
                    f"  Puan: {p.get('google_ortalama_puan','-')}\n"
                    f"  Yorum Sayısı: {p.get('google_toplam_yorum','-')}\n"
                    f"  Telefon: {p.get('google_telefon','-')}\n"
                    f"  Web Sitesi: {p.get('google_web_sitesi','-')}\n"
                    f"  Harita Linki: {p.get('maps_linki','-')}\n"
                    f"  Fotoğraflar: {fotos_text}"
                )
            prompt_with_data = f"{system_message_template}\n\nİstanbul'daki uygun mekanlar:\n" + "\n".join(place_info) + f"\nKullanıcı: {last_user_message}"

    else:
        movie_info = "\n".join([
            f"• {m.get('title','-')}, Türler: {', '.join(m.get('genres',[]))}, "
            f"IMDb: {m.get('imdb_rating','-')}, Oyuncular: {', '.join([a.get('name') for a in m.get('actors',[]) if a.get('name')][:3])}"
            for m in movies_data
        ])
        prompt_with_data = f"{system_message_template}\n\nFilm ve dizi verileri:\n{movie_info}\nKullanıcı: {last_user_message}"


    try:
        response = invoke_llm_with_timeout([HumanMessage(content=prompt_with_data)], timeout_seconds=120)
        content = response.content.strip() or "Üzgünüm, şu anda yanıt oluşturamıyorum."
        return {"messages":[AIMessage(content=content)]}
    except Exception as e:
        return {"messages":[AIMessage(content=f"Üzgünüm, yanıt oluşturulamadı: {e}")]}


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


# ----------------- LLM client -----------------

ollama_client = None

if ollama_client is None:
    try:
        ollama_client = get_ollama_client()
    except Exception as e:
        st.error(f"LLM client başlatılamadı: {e}")

# ----------------- Session state messages -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- Neo4j connector -----------------
neo4j_connector = Neo4jConnector()


for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, dict):
        if msg.get("role") == "user":
            st.session_state.messages[i] = HumanMessage(content=msg.get("content", ""))
        elif msg.get("role") == "assistant":
            st.session_state.messages[i] = AIMessage(content=msg.get("content", ""))

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
    safe_save_chat(neo4j_connector, session_id, "user", prompt, datetime.now())
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Yanıt oluşturuluyor..."):
        app = create_workflow()
        config = {"configurable": {"thread_id": session_id}}
        latest_ai_message_content = ""
        try:
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
                error_msg = "Üzgünüm, bir yanıt üretemedim."
                ai_error_message = AIMessage(content=error_msg)
                st.session_state.messages.append(ai_error_message)
                safe_save_chat(neo4j_connector, session_id, "assistant", error_msg, datetime.now())
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
        except Exception as e:
            error_message = f"Bir hata oluştu: {e}."
            st.error(f"Ana döngüde beklenmedik hata: {str(e)}")
            ai_error_message = AIMessage(content=error_message)
            st.session_state.messages.append(ai_error_message)
            safe_save_chat(neo4j_connector, session_id, "assistant", error_message, datetime.now())
            with st.chat_message("assistant"):
                st.markdown(error_message)
