import streamlit as st
import os
import requests
import json
from typing import List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# ---------------- Ortam değişkenleri ----------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
TOOLS_SERVER_URL = "http://localhost:8000"

# ---------------- Tool Tanımları ----------------
tools = requests.get(f"{TOOLS_SERVER_URL}/tools").json()

# ---------------- Agent Helpers ----------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], list]
    last_recommended_place: Optional[str]

system_message_template = """
Sen bir sohbet robotusun ve kullanıcıya film, dizi veya İstanbul'da mekan önerileri sunuyorsun.
Ayrıca rezervasyon yapma ve sinema bileti bulma gibi işlemler için araçları kullanabilirsin.
Tüm yanıtların Türkçe olacak.
---
Kullanılacak araçlar:
{tools_schema}
---
"""

def generate_claude_response_with_tools(state: AgentState) -> AgentState:
    agent_state = st.session_state.agent_state
    user_message_content = state['messages'][-1].content
    
    # Tool'ları Anthropic API'nin beklediği formata dönüştür
    tools_for_claude = [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": {
                "type": "object",
                "properties": {k: {"type": v["type"]} for k, v in tool.get("parameters", {}).items()},
                "required": list(tool.get("parameters", {}).keys())
            }
        } for tool in tools
    ]
    
    # Claude'a ilk mesajı gönder
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        },
        json={
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": user_message_content}],
            "tools": tools_for_claude
        }
    ).json()

    # Modelin yanıtını kontrol et
    if response.get("stop_reason") == "tool_use":
        tool_use = response["content"][0]
        tool_name = tool_use["name"]
        tool_input = tool_use["input"]
        
        # Tool'u çağır
        tool_response = requests.get(f"{TOOLS_SERVER_URL}/tools/{tool_name}", params=tool_input).json()

        # Tool'un sonucunu tekrar Claude'a gönder
        second_response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": user_message_content},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": tool_use["id"], "name": tool_name, "input": tool_input}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": json.dumps(tool_response)}]}
                ]
            }
        ).json()
        
        final_content = second_response["content"][0]["text"]
        agent_state['messages'].append(AIMessage(content=final_content))
    else:
        # Eğer model tool kullanmaya karar vermezse, doğrudan metin yanıtı ver
        final_content = response["content"][0]["text"]
        agent_state['messages'].append(AIMessage(content=final_content))
        
    return agent_state

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="İstanbul Chatbotu", layout="wide")
st.title("İstanbul Chatbotu: Film, Mekan ve Rezervasyon")

if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {'messages': []}

# Sohbet geçmişini göster
for msg in st.session_state.agent_state['messages']:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

user_input = st.chat_input("Sorunuzu yazın:")

if user_input:
    st.session_state.agent_state['messages'].append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Yanıtınız oluşturuluyor..."):
            updated_state = generate_claude_response_with_tools({'messages': [HumanMessage(content=user_input)]})
            
            for msg in updated_state['messages']:
                if isinstance(msg, AIMessage) and msg not in st.session_state.agent_state['messages']:
                    st.markdown(msg.content)