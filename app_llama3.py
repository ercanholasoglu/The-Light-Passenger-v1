# app_llama3.py
import streamlit as st
import time
from neo4j import GraphDatabase
from datetime import datetime
from typing import List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
import sys
import re
import os
import uuid
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import json
import random

# --------------------
# Ortam değişkenleri
# --------------------
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "llama3-8b-tr")
# Temperature sabit 0.7 (isteğin üzerine)
LLM_TEMPERATURE = 0.7

# --------------------
# Yardımcı fonksiyonlar
# --------------------
def sanitize_markdown(text):
    if not isinstance(text, str):
        return str(text)
    if not text:
        return ""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Çok küçük kaçış
    return text

# Güvenli lower kullanımı (None kontrolü)
def safe_lower(s):
    return s.lower() if isinstance(s, str) else ""

# --------------------
# Neo4j connector (aynı kaldı)
# --------------------
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

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

# --------------------
# Veri çekme (örnek sorgular)
# --------------------
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
    # Basit çekme; hata yakalama
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

# Load data at startup (graceful empty fallback)
movies_data = []
places_data = []
try:
    movies_data = fetch_all_movie_data_with_details(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATABASE, movie_query)
    places_data = fetch_all_place_data(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, "neo4j", places_query)
    print(f"✔️ Veri yüklendi. Filmler: {len(movies_data)}, Mekanlar: {len(places_data)}")
except Exception as e:
    print("Veri yüklenemedi:", e)

# --------------------
# LLM client başlatma (tek seferlik, global)
# --------------------
def get_ollama_client():
    # wrapper
    return ChatOllama(model=OLLAMA_MODEL_NAME, temperature=LLM_TEMPERATURE)

# Thread pool for timeouts
_executor = ThreadPoolExecutor(max_workers=2)

def invoke_llm_with_timeout(messages: List[BaseMessage], timeout_seconds: int = 30):
    client = st.session_state.get("ollama_client")
    if client is None:
        raise RuntimeError("LLM client başlatılmamış.")
    future = _executor.submit(client.invoke, messages)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeout:
        future.cancel()
        raise TimeoutError(f"LLM çağrısı {timeout_seconds}s içinde tamamlanmadı.")
    except Exception:
        raise

# --------------------
# Tool'lar (placeholder)
# - Model çağırmak için özel protokol: TOOL tag içinde JSON
#   <TOOL>{"name":"reservation_check","args":{...}}</TOOL>
# --------------------
def tool_check_reservation(original_id: Optional[int], requested_time: Optional[str]=None):
    """
    Gerçek uygulamada burada rezervasyon servisine istek atılacak.
    Şimdilik simüle ediyoruz: eğer original_id çiftse uygun, tekse dolu gibi rastgele mantık.
    Dönen yapı:
    { "ok": True, "available": True/False, "available_slots": [...], "book_url": "..." }
    """
    try:
        # Simülasyon: eğer verilen id yoksa hata bildir
        if original_id is None:
            return {"ok": False, "error": "place id yok."}
        # Basit simülasyon: rastgele slot üret
        slots = ["18:30", "19:30", "20:30", "21:30"]
        random.shuffle(slots)
        available_slots = slots[:2]
        # Simule edilecek "rezervasyon var mı" bilgisi
        available = True if random.random() > 0.2 else False
        return {
            "ok": True,
            "available": available,
            "available_slots": available_slots if available else [],
            "book_url": f"https://example.com/book?place_id={original_id}"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def tool_find_streaming_or_ticket(title: str):
    """
    Eğer film/dizi veritabanında varsa platform veya vizyona dair bilgi döndürür.
    Dönen yapı:
    { "ok": True, "in_theaters": True/False, "cinema_links": [...], "streaming_platforms": [...] }
    """
    try:
        if not title:
            return {"ok": False, "error": "title missing"}
        # Basit eşleme
        found = None
        for m in movies_data:
            if m.get("title") and m.get("title").strip().lower() == title.strip().lower():
                found = m
                break
        # Simülasyon: eğer bulunduysa, platform listesi döndür
        if found:
            platforms = found.get("platforms") or []
            # Simüle vizyonda durumu rastgele (gerçekte release date kontrolü gerekir)
            in_theaters = True if random.random() > 0.7 else False
            cinema_links = []
            if in_theaters:
                cinema_links = [f"https://example.com/tickets?movie={found.get('title').replace(' ','+')}"]
            streaming = [p.get("platform_name") for p in platforms if p.get("platform_name")]
            return {"ok": True, "in_theaters": in_theaters, "cinema_links": cinema_links, "streaming_platforms": streaming}
        else:
            # Bulunamadıysa genel arama simülasyonu
            return {"ok": True, "in_theaters": False, "cinema_links": [], "streaming_platforms": []}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# --------------------
# System prompt + model-tool protokolü
# --------------------
system_message_template = """
Sen İstanbul merkezli bir asistan/rehbersin. Kullanıcıya film/dizi veya İstanbul içi mekan önerileri sunacaksın.
- Tüm yanıtların **Türkçe** olacak.
- Mekan sorularında **sadece** Neo4j'deki `places_data` verisini kullan.
- Film/dizi sorularında **sadece** `movies_data` verisini kullan.
- Öneri yaparken **her zaman 5 tane** mekan öner (varsa semte göre filtrele; yoksa en popüler 5).
- Kullanıcıya gösterirken her bir mekanın aşağıdaki bilgilerini ver:
  - Adı
  - Adres
  - Fiyat seviyesi (fiyat_seviyesi_simge)
  - Puan (google_ortalama_puan)
  - Yorum sayısı
  - Telefon
  - Web sitesi
  - Maps linki
- Eğer model dış araç (tool) çağırması gerekirse, **mutlaka** aşağıdaki protokole uygun bir JSON bloğu üret:
  - Format: <TOOL>{...json...}</TOOL>
  - Örnek rezervasyon check çağrısı:
    <TOOL>{"name":"reservation_check", "args":{"original_id": 107, "requested_time": "2025-09-10 19:30"}}</TOOL>
  - Örnek film gösterim/tool çağrısı:
    <TOOL>{"name":"find_streaming_or_ticket", "args":{"title":"Inception"}}</TOOL>
- Uygulama bu JSON'u okuyacak, ilgili tool'u çalıştırıp sonucunu sana verecek; sonra sen sonuçları kullanıp nihai cevabı oluşturacaksın.
- Eğer tool çıktısına ihtiyaç yoksa tool çağırma.
- Cevaplarında uydurma bilgi (hallucination) verme; bilgin yoksa bunu açıkça söyle.
"""

# --------------------
# generate_response: LLM çağrısı + tool loop (basit)
# --------------------
def generate_response(user_message: str, user_id: Optional[str]=None, timeout_seconds: int=60):
    """
    user_message: istem (string)
    Dönen: assistant_text (str), tools_called (list)
    Mantık:
      - Semt tespiti -> 5 öneri hazırla -> prompt_with_data oluştur -> LLM çağır
      - LLM kullanıcının isteğine göre <TOOL>... çağırırsa parse ederek ilgili tool çalıştırılır
      - Tool çıktısı modele geri gönderilip modelin nihai cevabı alınır
    """
    # 1) Hazırla: semt tespiti
    lower = safe_lower(user_message)
    semt_keywords = ["suadiye", "kadıköy", "beşiktaş", "beyoğlu", "şişli", "ümraniye", "eyüpsultan", "göktürk"]
    user_semt = next((k for k in semt_keywords if k in lower), None)

    # 2) seçilecek 5 mekan
    selected_places = []
    if places_data:
        if user_semt:
            # semte uygun olanları al
            selected_places = [p for p in places_data if p.get("google_adres") and user_semt in p.get("google_adres","").lower()]
        # fallback: en yüksek puanlı top 5
        if not selected_places:
            # filtre None puan olanları çıkar
            with_score = [p for p in places_data if p.get("google_ortalama_puan") is not None]
            selected_places = sorted(with_score, key=lambda x: x.get("google_ortalama_puan",0), reverse=True)[:5]
    # Format et (5 tane olacak; eğer azsa mevcut kadar)
    selected_places = selected_places[:5]

    # 3) Place info text (model input)
    if selected_places:
        place_info_lines = []
        for p in selected_places:
            fotos = p.get("google_fotograf_linkleri") or []
            if fotos and all(isinstance(c, str) and len(c) == 1 for c in fotos):
                fotos = ["".join(fotos)]
            fotos_text = ", ".join(fotos[:3]) if fotos else "-"
            place_info_lines.append({
                "name": p.get("name") or "-",
                "google_adres": p.get("google_adres") or "-",
                "fiyat_seviyesi_simge": p.get("fiyat_seviyesi_simge") or "-",
                "google_ortalama_puan": p.get("google_ortalama_puan") or "-",
                "google_toplam_yorum": p.get("google_toplam_yorum") or "-",
                "google_telefon": p.get("google_telefon") or "-",
                "google_web_sitesi": p.get("google_web_sitesi") or "-",
                "maps_linki": p.get("maps_linki") or "-",
                "original_id": p.get("original_id"),
                "fotograflar": fotos_text
            })
        # prompt oluştur
        place_info_text = "\n\n".join([
            f"{i+1}) {pl['name']}\n"
            f"Adres: {pl['google_adres']}\n"
            f"Fiyat: {pl['fiyat_seviyesi_simge']}\n"
            f"Puan: {pl['google_ortalama_puan']}\n"
            f"Yorum Sayısı: {pl['google_toplam_yorum']}\n"
            f"Telefon: {pl['google_telefon']}\n"
            f"Web: {pl['google_web_sitesi']}\n"
            f"Maps: {pl['maps_linki']}\n"
            f"Fotoğraflar: {pl['fotograflar']}\n"
            f"original_id: {pl['original_id']}"
            for i,pl in enumerate(place_info_lines)
        ])
    else:
        place_info_text = "Veritabanında mekan bilgisi bulunamadı."

    # 4) Model çağrısı (ilk tur)
    system_msg = SystemMessage(content=system_message_template)
    prompt_with_data = f"{system_msg.content}\n\nMekan verileri:\n{place_info_text}\n\nKullanıcı: {user_message}"
    messages = [system_msg, HumanMessage(content=prompt_with_data)]

    try:
        # göstermek için UI tarafında "typing" görünümü kullanılacak (UI kısmı)
        raw_response = invoke_llm_with_timeout(messages, timeout_seconds=timeout_seconds)
    except Exception as e:
        return f"Üzgünüm, yanıt oluşturulamadı: {e}", []

    assistant_text = getattr(raw_response, "content", "") or ""
    tools_used = []

    # 5) Tool çağrısı kontrolü: model <TOOL>{...}</TOOL> üretmiş olabilir.
    # Parser: tüm <TOOL>...</TOOL> bloklarını bul ve JSON parse et.
    tool_blocks = re.findall(r"<TOOL>(.*?)</TOOL>", assistant_text, flags=re.DOTALL)
    # Eğer tool çağrısı varsa her biri için ilgili tool'u çağır ve sonucu modele gönder
    if tool_blocks:
        # Her block için parse & execute
        for block in tool_blocks:
            try:
                j = json.loads(block.strip())
            except Exception as e:
                # parse hatası -> modele bildirilecek
                tools_used.append({"raw": block, "error": f"JSON parse hatası: {e}"})
                continue

            tname = j.get("name")
            args = j.get("args", {})
            # Çalıştır
            if tname == "reservation_check":
                original_id = args.get("original_id")
                requested_time = args.get("requested_time")
                out = tool_check_reservation(original_id=original_id, requested_time=requested_time)
                tools_used.append({"tool": tname, "args": args, "output": out})
                # Tool çıktısını modele tekrar gönder: formata uygun bir mesaj oluştur
                tool_msg = SystemMessage(content=f"TOOL_OUTPUT {json.dumps({'tool':tname,'output':out}, ensure_ascii=False)}")
                # Modeli tekrar çağır ve son cevabı al
                try:
                    follow = invoke_llm_with_timeout([system_msg, HumanMessage(content=prompt_with_data), tool_msg], timeout_seconds=timeout_seconds)
                    assistant_text = getattr(follow, "content", "") or assistant_text
                except Exception as e:
                    assistant_text += f"\n\n(Not: tool sonrası LLM çağrısı başarısız: {e})"
            elif tname == "find_streaming_or_ticket":
                title = args.get("title")
                out = tool_find_streaming_or_ticket(title=title)
                tools_used.append({"tool": tname, "args": args, "output": out})
                tool_msg = SystemMessage(content=f"TOOL_OUTPUT {json.dumps({'tool':tname,'output':out}, ensure_ascii=False)}")
                try:
                    follow = invoke_llm_with_timeout([system_msg, HumanMessage(content=prompt_with_data), tool_msg], timeout_seconds=timeout_seconds)
                    assistant_text = getattr(follow, "content", "") or assistant_text
                except Exception as e:
                    assistant_text += f"\n\n(Not: tool sonrası LLM çağrısı başarısız: {e})"
            else:
                tools_used.append({"tool": tname, "error": "Unknown tool"})
    # Eğer model Türkçe dışında cevap verdiyse, zorla Türkçeye çevirme turu (ek LLM çağrısı)
    if not re.search(r"[ığüşöçİĞÜŞÖÇ]", assistant_text):  # kaba kontrol: türkçe karakter yoksa muhtemelen ingilizce
        # isteği Türkçeleştir talebi gönder
        correction_prompt = SystemMessage(content="Lütfen yukarıdaki yanıtı tamamen Türkçe olarak yeniden ver.")
        try:
            follow = invoke_llm_with_timeout([system_msg, HumanMessage(content=assistant_text), correction_prompt], timeout_seconds=30)
            assistant_text = getattr(follow, "content", "") or assistant_text
        except Exception:
            # başarısız olursa orijinali dön
            pass

    return assistant_text.strip(), tools_used

# --------------------
# Streamlit UI: sohbet geçmişi, typing, spinner, enter ile submit
# --------------------
st.set_page_config(page_title="İstanbul Chatbotu - Film & Mekan Önerileri", layout="wide")
st.title("İstanbul Chatbotu - Film & Mekan Önerileri")

# init session state
if "messages" not in st.session_state:
    # messages: list of dicts {role: 'user'/'assistant', content: str}
    st.session_state.messages = []
if "ollama_client" not in st.session_state:
    try:
        st.session_state.ollama_client = get_ollama_client()
    except Exception as e:
        st.error(f"LLM client başlatılamadı: {e}")

# Chat history gösterimi (eski alt alta format, her yeni mesaj appended)
def render_chat_history():
    for msg in st.session_state.messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            st.markdown(f"**Siz:** {sanitize_markdown(content)}")
        else:
            # assistant
            st.markdown(f"**Bot:** {sanitize_markdown(content)}")

# Sol sütunda chat, sağda opsiyonel detay (basit layout)
col1, col2 = st.columns([3,1])

with col1:
    render_chat_history()
    placeholder = st.empty()
    # input form (Enter ile submit)
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Sorunuzu yazın (ör. 'Suadiye'de date mekanı öner')", key="user_input")
        submitted = st.form_submit_button("Gönder")
        if submitted and user_input:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            # show "typing" placeholder
            typing_slot = placeholder.container()
            with typing_slot:
                st.info("Yanıtınız oluşturuluyor... Lütfen bekleyin.")
            # Generate response (LLM çağrısı + tool loop)
            try:
                assistant_text, tools_used = generate_response(user_input, user_id=str(uuid.uuid4()), timeout_seconds=60)
            except Exception as e:
                assistant_text = f"Üzgünüm, yanıt oluşturulamadı: {e}"
                tools_used = []
            # Replace typing with real assistant message: append to history and rerender
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            # clear typing slot by re-rendering history
            typing_slot.empty()
            # Rerender entire history (simple approach)
            placeholder.empty()
            render_chat_history()
            # Eğer tool kullanıldıysa kısa özet göster
            if tools_used:
                st.markdown("---")
                st.markdown("**Tool çağrıları:**")
                st.json(tools_used)

with col2:
    st.markdown("### Bilgiler")
    st.markdown(f"- Mekan sayısı (DB): {len(places_data)}")
    st.markdown(f"- Film/dizi sayısı (DB): {len(movies_data)}")
    st.markdown("- Temperature: 0.7 (sabit)")
    st.markdown("---")
    st.markdown("Tool protokolü örneği:")
    st.code('''<TOOL>{"name":"reservation_check","args":{"original_id":107,"requested_time":"2025-09-10 19:30"}}</TOOL>''')
    st.code('''<TOOL>{"name":"find_streaming_or_ticket","args":{"title":"Inception"}}</TOOL>''')
