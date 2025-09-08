import streamlit as st
import time
import re
import os
from neo4j import GraphDatabase
from datetime import datetime
from typing import List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import queue
import threading

# ---------------- Ortam değişkenleri ----------------
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "llama3-8b-tr")
DATABASE_MOVIE = "moviesandseries"
DATABASE_PLACE = "neo4j"

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
                self.driver.verify_connectivity()
            except Exception as e:
                st.error(f"Neo4j veritabanına bağlanılamadı. Hata: {e}")
                st.stop()

    def fetch(self, query):
        self.connect()
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None
            
# ---------------- LLM Client ----------------
def get_ollama_client():
    return ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.7)

# ---------------- Veri Çekme Fonksiyonları (Önbellekleme ile) ----------------

@st.cache_data(ttl=3600)
def fetch_movies_data():
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
            .title, .imdb_id, .imdb_rating, .imdb_votes, .metacritic,
            .box_office, .awards, .rotten_tomatoes, .homepage, .budget,
            .revenue, .vote_average, .vote_count, .runtime, .poster,
            .rated, .overview
        } AS movie_props, 
        collect(DISTINCT {name: l.name, code: l.code}) AS languages, 
        collect(DISTINCT g.name) AS genres, 
        collect(DISTINCT co.name) AS countries, 
        collect(DISTINCT comp.name) AS companies, 
        collect(DISTINCT {name: a.name, character: rel_acted.character, order: rel_acted.order}) AS actors,
        collect(DISTINCT {name: cr.name, job: rel_worked.job, department: rel_worked.department}) AS crew,
        collect(DISTINCT {platform_name: p.name, country: rel_avail.country, type: rel_avail.type}) AS platforms
    LIMIT 200
    """
    neo = Neo4jConnector(DATABASE_MOVIE)
    data = neo.fetch(movie_query)
    neo.close()
    movies = []
    for record in data:
        m = dict(record['movie_props'])
        m['genres'] = record['genres']
        m['languages'] = record['languages']
        m['countries'] = record['countries']
        m['companies'] = record['companies']
        m['actors'] = record['actors']
        m['crew'] = record['crew']
        m['platforms'] = record['platforms']
        movies.append(m)
    return movies

@st.cache_data(ttl=3600)
def fetch_places_data():
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
    LIMIT 200
    """
    neo = Neo4jConnector(DATABASE_PLACE)
    data = neo.fetch(places_query)
    neo.close()
    return data

# Verileri önceden yükle
movies_data = fetch_movies_data()
places_data = fetch_places_data()

# ---------------- Agent Helpers ----------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], list]
    last_recommended_place: Optional[str]

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

def generate_llm_response(state: AgentState) -> AgentState:
    agent_state = st.session_state.agent_state
    user_message = state['messages'][-1].content
    
    last_user_lower = (user_message or "").lower()
    selected_places = []
    data_context = ""

    is_place_request = any(k in last_user_lower for k in ["mekan","yer","restoran","bar","cafe","randevu","date"])
    
    if is_place_request and places_data:
        semt_keywords = ["suadiye","kadıköy","beşiktaş","beyoğlu","şişli","ümraniye","eyüpsultan","göktürk"]
        user_semt = next((k for k in semt_keywords if k in last_user_lower), None)

        if user_semt:
            selected_places = [p for p in places_data if (p.get('google_adres') or "").lower().find(user_semt) >= 0]
        else:
            selected_places = sorted(places_data, key=lambda x: x.get('google_ortalama_puan',0), reverse=True)[:5]

        place_info = []
        for p in selected_places[:5]:
            fotos = p.get('google_fotograf_linkleri') or []
            fotos_text = ", ".join(fotos[:3]) if fotos else "-"
            place_info.append(
                f"{p.get('name','-')}\n"
                f"Adres: {p.get('google_adres','-')}\n"
                f"Fiyat: {p.get('fiyat_seviyesi_simge','-')}\n"
                f"Puan: {p.get('google_ortalama_puan','-')}\n"
                f"Yorum Sayısı: {p.get('google_toplam_yorum','-')}\n"
                f"Telefon: {p.get('google_telefon','-')}\n"
                f"Web: {p.get('google_web_sitesi','-')}\n"
                f"Maps: {p.get('maps_linki','-')}\n"
                f"Fotoğraflar: {fotos_text}\n"
            )
        data_context = "\n\nİstanbul'daki uygun mekanlar:\n" + "\n".join(place_info)

    else:
        movie_info = "\n".join([
            f"{m.get('title','-')}, Türler: {', '.join(m.get('genres',[]))}, IMDb: {m.get('imdb_rating','-')}, "
            f"Oyuncular: {', '.join([a.get('name') for a in m.get('actors',[]) if a.get('name')][:3])}"
            for m in movies_data[:5]
        ])
        data_context = "\n\nFilm ve dizi verileri:\n" + movie_info
    
    # LLM'e gönderilecek prompt'u oluştur
    full_prompt = f"{system_message_template}\n\n{data_context}\n\nKullanıcı: {user_message}"
    
    try:
        response = st.session_state.ollama_client.invoke([HumanMessage(content=full_prompt)], temperature=0.7)
        content_clean = re.sub(r"<TOOL>.*?</TOOL>", "", response.content, flags=re.DOTALL).strip()
        if not content_clean:
            content_clean = "Üzgünüm, şu anda yanıt oluşturamıyorum."
        agent_state['messages'].append(AIMessage(content=content_clean))
        return agent_state
    except Exception as e:
        agent_state['messages'].append(AIMessage(content=f"Üzgünüm, yanıt oluşturulamadı: {e}"))
        return agent_state

# ---------------- Streamlit App ----------------
if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {'messages': [], 'last_recommended_place': None}
if 'ollama_client' not in st.session_state:
    st.session_state.ollama_client = get_ollama_client()

st.set_page_config(page_title="İstanbul Chatbotu", layout="wide")
st.title(" İstanbul Chatbotu: Film & Mekan Önerileri")
st.write("Film, dizi veya İstanbul'da mekan önerileri almak için bir soru sorun.")

# Sohbet geçmişini göster
for msg in st.session_state.agent_state['messages']:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**Siz:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**Bot:** {msg.content}")

user_input = st.chat_input("Sorunuzu yazın:")

if user_input:
    # Kullanıcı mesajını geçmişe ekle
    st.session_state.agent_state['messages'].append(HumanMessage(content=user_input))
    
    # Yanıt oluşturulduğunu göster
    with st.spinner("Yanıtınız oluşturuluyor..."):
        # LLM'den yanıtı al
        updated_state = generate_llm_response({'messages': [HumanMessage(content=user_input)]})
    
    # Yeni mesajı göster
    for msg in updated_state['messages']:
        if isinstance(msg, AIMessage) and msg not in st.session_state.agent_state['messages']:
            st.markdown(f"**Bot:** {msg.content}")