import os
import re
import json
import requests
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple

# LangChain + Ollama (mevcut kullanım korunur)
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# Neo4j
from neo4j import GraphDatabase

# .env
from dotenv import load_dotenv
load_dotenv()  # .env dosyası varsa yükle

# =========================
# ENV & SABİTLER
# =========================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "my_tr_llama3")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")  # MCP server (FastAPI)

APP_TITLE = "The Light Passenger"

# =========================
# yardımcılar
# =========================
@st.cache_resource(show_spinner=False)
def get_neo4j_driver(uri: str, user: str, password: str):
    """Cache'li Neo4j driver (app ömrünce 1 kez)."""
    if not uri or not user:
        raise RuntimeError("NEO4J_URI ve NEO4J_USER env değişkenleri zorunludur.")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    # hızlı doğrulama (destek yoksa sessiz geçer)
    try:
        driver.verify_connectivity()
    except Exception:
        pass
    return driver

def run_cypher(query: str, params: Optional[Dict[str, Any]] = None, db: Optional[str] = None) -> List[Dict[str, Any]]:
    """Cypher çalıştır, dict listesi döndür."""
    driver = get_neo4j_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    params = params or {}
    with driver.session(database=db) as session:
        result = session.run(query, **params)
        return [r.data() for r in result]

@st.cache_data(ttl=60, show_spinner=False)
def get_counts() -> Tuple[int, int]:
    """
    Movies & Series toplamı ile Meyhane sayısını getirir.
    - Movie + Series toplamı (varsayılan DB: moviesandseries olduğu varsayılır)
    - Meyhane sayısı   (varsayılan DB: neo4j olduğu varsayılır)
    DB isimlerini kodda sabitlemek yerine, iki sorguyu arka arkaya default db'de çalıştırıyoruz.
    İstersen bunları farklı db'lerde çağırmak için session(database=...) paramı ekleyebilirsin.
    """
    # Movie + Series toplamı
    q_total = """
    CALL {
      MATCH (m:Movie) RETURN count(m) AS c
    }
    CALL {
      MATCH (s:Series) RETURN count(s) AS d
    }
    RETURN c + d AS total
    """
    # Meyhane sayısı
    q_meyhane = "MATCH (p:Meyhane) RETURN count(p) AS cnt"

    try:
        total = run_cypher(q_total)[0]["total"]
    except Exception:
        total = 0
    try:
        meyhane = run_cypher(q_meyhane)[0]["cnt"]
    except Exception:
        meyhane = 0
    return int(total), int(meyhane)

def mask_value(val: Optional[str]) -> str:
    if not val:
        return "(not set)"
    return (val[:3] + "...") if len(val) > 3 else "xxx"

# =========================
# MCP / TOOL ÇAĞRI YARDIMCILARI
# =========================
TOOL_ENDPOINTS: Dict[str, Dict[str, Any]] = {
    # örnek: get_movies aracı
    # expected input: {"rotten_tomatoes_score": "90"} ya da "90.0"
    "get_movies": {
        "method": "GET",
        "path": "/tools/get_movies",
        "map": lambda tool_input: {
            "rotten_threshold": float(tool_input.get("rotten_tomatoes_score", 90))
        },
    },
    # buraya yeni tool'lar eklersin
}

def parse_tool_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Kullanıcının/assistanın ürettiği yanıttan tool_code bloğunu yakalar.
    Format beklenen:
    </tool_code>
    { "tool_name": "...", "tool_input": {...} }
    </tool_code>
    """
    pattern = re.compile(r"</tool_code>\s*(\{.*?\})\s*</tool_code>", re.DOTALL)
    m = pattern.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None

def call_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """MCP FastAPI'ye tool çağrısı yapar."""
    if tool_name not in TOOL_ENDPOINTS:
        return {"error": f"Bilinmeyen tool: {tool_name}"}
    spec = TOOL_ENDPOINTS[tool_name]
    url = FASTAPI_URL.rstrip("/") + spec["path"]
    params_or_json = spec["map"](tool_input)

    try:
        if spec["method"].upper() == "GET":
            r = requests.get(url, params=params_or_json, timeout=20)
        else:
            r = requests.post(url, json=params_or_json, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"Araç çağrısı başarısız: {e}"}

# =========================
# OLLAMA / AGENT
# =========================
def get_llm():
    # LangChain 0.3.1 sonrası uyarı alabilirsin; ancak mevcut kullanım korunuyor
    # Alternatif: from langchain_ollama import ChatOllama (paketi ayrıca yüklemek gerekir).
    return ChatOllama(model=OLLAMA_MODEL_NAME)

SYSTEM_PROMPT = (
    "Sen İstanbul odaklı bir yardımcı asistansın. Kullanıcı film, mekan (Meyhane), rezervasyon ve benzeri konularda soru sorabilir. "
    "Uygunsa veritabanı araçlarını kullan. Araç kullanman gerekiyorsa, aşağıdaki formatta yalnızca tek bir tool çağrısı oluştur:\n"
    "</tool_code>\n"
    "{ \"tool_name\": \"get_movies\", \"tool_input\": { \"rotten_tomatoes_score\": \"90\" } }\n"
    "</tool_code>\n"
    "Eğer araç gerekli değilse normal yanıt ver. Türkçe konuş."
)

def generate_response_with_tools(user_message: str) -> str:
    """
    1) LLM'e soruyu gönder (kılavuz prompt ile).
    2) Eğer tool_code bloğu dönerse ilgili MCP Tool'u çağır ve sonuçları kullanıcıya yaz.
    3) Aksi halde LLM'in ürettiği metni döndür.
    """
    llm = get_llm()
    try:
        msg = llm.invoke([HumanMessage(content=f"{SYSTEM_PROMPT}\n\nKullanıcı: {user_message}")])
        text = msg.content if isinstance(msg, AIMessage) else str(msg)
    except Exception as e:
        return f"Model çağrısı sırasında hata oluştu: {e}"

    tool_req = parse_tool_block(text)
    if tool_req:
        tool_name = tool_req.get("tool_name")
        tool_input = tool_req.get("tool_input", {})
        tool_result = call_tool(tool_name, tool_input)
        if "error" in tool_result:
            return f"Araç çağrısı sırasında bir hata oluştu: {tool_result['error']}"
        # Tool spesifik cevap biçimi
        if tool_name == "get_movies":
            movies = tool_result.get("movies", [])
            if not movies:
                return "Bu kritere uygun film bulunamadı."
            lines = ["**Önerilen Filmler:**"]
            for m in movies:
                title = m.get("title", "Unknown")
                year = m.get("year", "")
                rt = m.get("rt", "")
                lines.append(f"- {title} ({year}) — RT: {rt}")
            return "\n".join(lines)
        # diğer tool'lar için format…
        return f"Araç sonucu: {json.dumps(tool_result, ensure_ascii=False, indent=2)}"
    else:
        # normal yanıtı geri ver
        return text

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.markdown("### Helpful Links :)")
    st.markdown("- [LinkedIn](https://www.linkedin.com/in/ercan-holasoğlu1/)")
    st.markdown("- [Kaggle](https://www.kaggle.com/ercanholasoglu)")
    st.markdown("- [HuggingFace](https://huggingface.co/SutskeverFanBoy)")
    st.markdown("- [GitHub](https://github.com/ercanholasoglu)")
    st.markdown("---")
    st.markdown("**Env Önizleme (masked)**")
    st.json({
        "NEO4J_URI": os.getenv("NEO4J_URI", "(not set)"),
        "NEO4J_USER": os.getenv("NEO4J_USER", "(not set)"),
        "NEO4J_PASSWORD": mask_value(os.getenv("NEO4J_PASSWORD")),
        "OLLAMA_MODEL_NAME": os.getenv("OLLAMA_MODEL_NAME", "(not set)"),
        "FASTAPI_URL": os.getenv("FASTAPI_URL", "(not set)")
    })

st.title(APP_TITLE)
st.subheader("İstanbul Chatbotu: Film, Mekan ve Rezervasyon")

# Üst metrikler
col_m1, col_m2, _ = st.columns([1, 1, 2])
with col_m1:
    with st.spinner("Toplam Movie+Series sayısı hesaplanıyor..."):
        total_ms, _mey_dummy = get_counts()
    st.metric("🎬 Movies & Series Count", f"{total_ms:,}")

with col_m2:
    with st.spinner("Meyhane sayısı hesaplanıyor..."):
        _total_dummy, meyhane_cnt = get_counts()
    st.metric("🍷 Meyhane Count", f"{meyhane_cnt:,}")

st.divider()

# Ana layout
left, right = st.columns([2, 1])

with left:
    st.header("Sohbet")
    # Chat geçmişi
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Önceki mesajları göster
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Sorunuzu yazın:")
    if user_input:
        # kullanıcı mesajını ekrana ve geçmişe yaz
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Asistan cevabı
        with st.chat_message("assistant"):
            with st.spinner("Yanıt oluşturuluyor..."):
                response = generate_response_with_tools(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

with right:
    st.header("Hızlı Araçlar (MCP)")
    st.caption("Aşağıdaki butonlar test amaçlı MCP FastAPI uçlarını çağırır.")

    # get_movies testi
    rotten = st.slider("Minimum RottenTomatoes (%)", 0, 100, 90, 1, key="rt_slider")
    if st.button("🔧 get_movies çağır"):
        try:
            r = requests.get(
                f"{FASTAPI_URL.rstrip('/')}/tools/get_movies",
                params={"rotten_threshold": rotten, "limit": 50},
                timeout=15
            )
            r.raise_for_status()
            data = r.json()
            movies = data.get("movies", [])
            if not movies:
                st.info("Sonuç bulunamadı.")
            else:
                for m in movies:
                    title = m.get("title", "Unknown")
                    year = m.get("year", "")
                    rt = m.get("rt", "")
                    st.write(f"- **{title}** ({year}) — RT: **{rt}**")
        except Exception as e:
            st.error(f"Araç çağrısı başarısız: {e}")

st.markdown("---")
st.caption("Not: Eğer FastAPI MCP sunucusu çalışmıyorsa, sağdaki MCP butonları hata verebilir. "
           "FastAPI için: `uvicorn MCP_For_v1:app --reload` veya kendi servis dosyanı çalıştır." )
