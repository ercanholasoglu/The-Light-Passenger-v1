import streamlit as st
import os
import requests
import json
from typing import List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
import re
from neo4j import GraphDatabase
import sys

# ---------------- Ortam değişkenleri ----------------
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "llama3-8b-tr")
TOOLS_SERVER_URL = "http://localhost:8000"

# ---------------- Ollama Client ve Araç Tanımları ----------------
@st.cache_resource
def get_ollama_client():
    try:
        # Streamlit'in kendi önbellekleme mekanizması
        return ChatOllama(model=OLLAMA_MODEL_NAME)
    except Exception as e:
        st.error(f"Ollama modeli başlatılamadı. Hata: {e}. Modelin çalıştığından emin olun.")
        st.stop()
        return None

try:
    tools_response = requests.get(f"{TOOLS_SERVER_URL}/tools")
    tools_response.raise_for_status()
    tools_data = tools_response.json()
except requests.exceptions.RequestException as e:
    st.error(f"Araç sunucusuna (FastAPI) bağlanılamadı. Lütfen sunucunun çalıştığından emin olun. Hata: {e}")
    st.stop()

# LLM'ye araçları tanıtmak için sistem mesajı oluşturma
def create_system_message_with_tools(tools_list):
    tool_descriptions = "\n".join([
        f"- {t['name']}: {t['description']}" for t in tools_list
    ])
    
    tool_schemas = json.dumps(tools_list, indent=2, ensure_ascii=False)
    
    return f"""
Sen bir sohbet robotusun ve kullanıcıya film, mekan veya rezervasyon önerileri sunuyorsun.
Bu işlemleri yapabilmek için aşağıdaki araçları kullanabilirsin.
Bir araç kullanman gerektiğinde, sadece JSON formatında aşağıdaki gibi yanıt ver:

<tool_code>
{{
  "tool_name": "arac_adi",
  "tool_input": {{
    "param1": "deger1",
    "param2": "deger2"
  }}
}}
</tool_code>

Ardından, kullanıcının sorusuna uygun ve detaylı bir yanıt ver.

Mevcut araçlar:
{tool_schemas}
"""

# Ollama ile yanıt oluşturma fonksiyonu
def generate_response_with_tools(user_message):
    ollama_client = get_ollama_client()
    system_message = create_system_message_with_tools(tools_data)
    
    # 1. Aşama: Modelden araç çağrısı iste
    prompt_for_tool_call = f"{system_message}\n\nKullanıcı: {user_message}"
    response = ollama_client.invoke(prompt_for_tool_call)
    
    full_content = response.content
    tool_call_match = re.search(r'<tool_code>(.*?)</tool_code>', full_content, re.DOTALL)
    
    if tool_call_match:
        try:
            tool_json_str = tool_call_match.group(1)
            tool_call_data = json.loads(tool_json_str)
            tool_name = tool_call_data.get("tool_name")
            tool_input = tool_call_data.get("tool_input", {})
            
            # 2. Aşama: Aracı çağır ve sonucu al
            st.info(f"Araç kullanılıyor: {tool_name}...")
            tool_response = requests.get(f"{TOOLS_SERVER_URL}/tools/{tool_name}", params=tool_input).json()
            
            # 3. Aşama: Aracı cevabını modele geri besle
            prompt_with_tool_result = f"{system_message}\n\nKullanıcı: {user_message}\n\nAraç Sonucu:\n```json\n{json.dumps(tool_response, ensure_ascii=False)}\n```\n\nBu bilgiyi kullanarak kullanıcıya yanıt ver."
            final_response = ollama_client.invoke(prompt_with_tool_result)
            return final_response.content
            
        except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
            return f"Araç çağrısı sırasında bir hata oluştu: {e}"
    else:
        # Eğer model araç çağırmazsa, doğrudan yanıtı döndür
        return full_content

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="İstanbul Chatbotu", layout="wide")
st.title("İstanbul Chatbotu: Film, Mekan ve Rezervasyon")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sohbet geçmişini göster
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

user_input = st.chat_input("Sorunuzu yazın:")

if user_input:
    # Kullanıcı mesajını geçmişe ekle
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Yanıtınız oluşturuluyor..."):
            response_content = generate_response_with_tools(user_input)
            
            # Asistanın yanıtını geçmişe ekle ve göster
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            st.markdown(response_content)