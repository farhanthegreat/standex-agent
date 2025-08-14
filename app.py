import os
import json
from typing import List, Dict, Tuple

import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process as rf_process

# Optional OpenAI (only if key exists)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ------------- Utility: Load KB -------------
@st.cache_data(show_spinner=False)
def load_kb(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

KB_PATH = os.path.join("kb", "standex_kb.json")
KB = load_kb(KB_PATH)

LANGS = ["bn", "en", "ja"]

# ------------- Vectorizer for RAG -------------
@st.cache_resource(show_spinner=False)
def build_vectorizer(kb: List[Dict], lang: str):
    docs = [doc.get(lang) or doc.get("en", "") for doc in kb]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(docs)
    return vec, X

# ------------- Retrieval -------------
def retrieve(query: str, kb: List[Dict], lang: str, top_k: int = 3) -> List[Tuple[float, Dict]]:
    vec, X = build_vectorizer(kb, lang)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    pairs = sorted([(float(sims[i]), kb[i]) for i in range(len(kb))], key=lambda x: x[0], reverse=True)
    return pairs[:top_k]

# ------------- Fuzzy intent by tags -------------
def fuzzy_tag_search(query: str, kb: List[Dict]) -> Dict:
    tags = []
    for item in kb:
        for t in item.get("tags", []):
            tags.append((t, item))
    choices = [t for t, _ in tags]
    if not choices:
        return {}
    match = rf_process.extractOne(query, choices, score_cutoff=70)
    if match:
        tag, score, index = match[0], match[1], match[2]
        return tags[index][1]
    return {}

# ------------- OpenAI helper -------------
def openai_answer(system_prompt: str, user_prompt: str, lang: str) -> str:
    if not OPENAI_API_KEY or OpenAI is None:
        return ""
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt + f"\n\nPlease answer in language: {lang}"}
    ]
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return ""

# ------------- Local fallback answer -------------
def local_compose_answer(query: str, kb_hits: List[Tuple[float, Dict]], lang: str) -> str:
    # If exact tag match exists, prioritize it
    tag_doc = fuzzy_tag_search(query, KB)
    if tag_doc:
        return tag_doc.get(lang) or tag_doc.get("en", "")

    # Otherwise, use the top RAG snippet
    if kb_hits:
        top = kb_hits[0][1]
        return top.get(lang) or top.get("en", "")

    # Default fallback
    if lang == "bn":
        return "দুঃখিত, আমি বিষয়টি বুঝতে পারিনি। অনুগ্রহ করে একটু বিস্তারিত বলবেন?"
    if lang == "ja":
        return "すみません、理解できませんでした。もう少し詳しく教えてください。"
    return "Sorry, I didn't catch that. Could you provide more details?"

# ------------- UI Helpers -------------
PRIMARY = "#0ea5e9"  # tailwind sky-500

def header():
    st.set_page_config(page_title="Standex Student Assistant", page_icon="🎌")
    left, mid, right = st.columns([0.15,0.6,0.25])
    with left:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.image("assets/logo.png", width=72, caption="Standex") if os.path.exists("assets/logo.png") else None
    with mid:
        st.markdown("""
        <h1 style='margin-bottom:0'>Standex Student Assistant</h1>
        <div style='color:#64748b'>Your bilingual helpdesk for Japanese language learning</div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:right'>Model: <b>{OPENAI_MODEL if OPENAI_API_KEY else 'Local RAG'}</b></div>", unsafe_allow_html=True)
    st.divider()

def quick_actions(lang: str):
    st.markdown("### Quick Actions")
    cols = st.columns(4)
    labels = {
        "bn": ["ভর্তি", "ফি", "ক্লাস সময়", "যোগাযোগ"],
        "en": ["Admission", "Fees", "Class Time", "Contact"],
        "ja": ["入学", "授業料", "時間割", "連絡先"],
    }
    prompts = ["admission", "fees", "class schedule", "contact"]
    for i, col in enumerate(cols):
        with col:
            if st.button(labels.get(lang, labels["en"])[i], use_container_width=True):
                st.session_state.chat.append({"role": "user", "content": prompts[i]})

# ------------- Main App -------------

def main():
    header()

    # Sidebar settings
    with st.sidebar:
        st.subheader("Settings")
        lang_choice = st.selectbox("Response language", ["Auto (Detect)", "Bengali (bn)", "English (en)", "Japanese (ja)"])
        temperature = st.slider("Creativity (if using OpenAI)", 0.0, 1.0, 0.2, 0.1)
        st.caption("Tip: No API key? The app still works with local answers.")
        st.markdown("---")
        st.caption("KB file: kb/standex_kb.json")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Quick actions
    sample_lang = "en"
    default_lang = "en"

    st.markdown("<div style='margin-top:-10px'></div>", unsafe_allow_html=True)

    # Determine UI language for labels (not for LLM)
    ui_lang = default_lang

    # Chat form
    with st.container():
        quick_actions(default_lang)
        st.markdown("### Ask anything")
        with st.form(key="chat_form", clear_on_submit=True):
            user_query = st.text_input("Type your question…", placeholder="e.g., ভর্তি প্রক্রিয়া কী? / How to enroll? / 入学方法は？")
            submitted = st.form_submit_button("Send")

    if submitted and user_query.strip():
        st.session_state.chat.append({"role": "user", "content": user_query})