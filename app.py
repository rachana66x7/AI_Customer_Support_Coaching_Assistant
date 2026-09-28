import os
import re
import json
import sqlite3
import hashlib
import hmac
import secrets
import io
import asyncio
from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st
from groq import Groq
from langdetect import detect
import edge_tts

# ============================================================
# CONFIG
# ============================================================
APP_NAME = "AI Customer Support Assistant"
DB_PATH = "customer_support_assistant.db"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
WHISPER_MODEL = "whisper-large-v3-turbo"

LANGUAGES = {
    "Auto Detect": "auto",
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Bengali": "bn",
    "Marathi": "mr",
}
LANGUAGE_NAMES = {v: k for k, v in LANGUAGES.items() if v != "auto"}

KB = [
    {
        "category": "Payment / Billing",
        "title": "Payment deducted but order failed",
        "keywords": ["payment", "deducted", "charged", "order failed", "billing", "transaction"],
        "answer": "Verify the payment and order status. Refund or failed-payment cases may require Accounts/Refund Team review."
    },
    {
        "category": "Order & Delivery",
        "title": "Delivery delay",
        "keywords": ["order", "delivery", "delayed", "late", "shipment", "tracking", "package"],
        "answer": "Check the latest tracking status and expected delivery date, then communicate the next step clearly."
    },
    {
        "category": "Technical Issue",
        "title": "Internet or application not working",
        "keywords": ["internet", "wifi", "network", "login", "app", "error", "technical", "connection"],
        "answer": "Guide the customer through simple troubleshooting steps and escalate when specialized support is required."
    },
    {
        "category": "Product Inquiry",
        "title": "Product information",
        "keywords": ["product", "price", "feature", "specification", "available", "availability", "model"],
        "answer": "Provide relevant product information and explain available options clearly."
    },
    {
        "category": "Refund / Cancellation",
        "title": "Refund or cancellation request",
        "keywords": ["refund", "cancel", "cancellation", "money back", "return", "reversal"],
        "answer": "Refund and cancellation requests may need transaction verification and specialized human handling."
    },
    {
        "category": "General Query",
        "title": "General support question",
        "keywords": ["question", "help", "how", "when", "where", "what", "information"],
        "answer": "Answer the customer's question concisely and confirm whether further assistance is needed."
    },
]

# ============================================================
# PAGE / STYLE
# ============================================================
st.set_page_config(page_title=APP_NAME, page_icon="🤝", layout="wide")
st.markdown("""
<style>
.stApp { background:#f4f7fb; }
.block-container { max-width:1500px; padding-top:1rem; }
#MainMenu, footer { visibility:hidden; }
.topbar { background:linear-gradient(110deg,#143a63,#25679d,#4e88c4); color:white; border-radius:18px; padding:20px 24px; margin-bottom:15px; box-shadow:0 8px 25px rgba(20,58,99,.15); }
.brand { font-size:29px; font-weight:800; }
.tagline { font-size:13px; opacity:.9; margin-top:4px; }
.panel { background:white; border:1px solid #e3e9f1; border-radius:15px; padding:16px; margin-bottom:14px; box-shadow:0 4px 14px rgba(15,23,42,.035); }
.panel-title { color:#163b60; font-size:13px; font-weight:850; letter-spacing:.06em; text-transform:uppercase; margin-bottom:10px; }
.metric-card { background:white; border:1px solid #e3e9f1; border-radius:14px; padding:14px; }
.metric-label { font-size:11px; color:#73849a; font-weight:800; text-transform:uppercase; letter-spacing:.05em; }
.metric-value { font-size:24px; font-weight:800; color:#163b60; margin-top:3px; }
.customer { background:#fff4f5; border-left:4px solid #eb707b; border-radius:10px; padding:11px; margin:7px 0; }
.agent { background:#eef6ff; border-left:4px solid #4f86df; border-radius:10px; padding:11px; margin:7px 0; }
.reply { background:#f0fdf4; border:1px solid #b8e5c5; border-radius:11px; padding:14px; line-height:1.6; }
.coach { background:#eff6ff; border:1px solid #c8d9f3; border-radius:11px; padding:14px; line-height:1.55; }
.handoff { background:#fffaf0; border:1px solid #eed28f; border-radius:11px; padding:14px; line-height:1.55; }
.decision-ai { background:#ecfdf3; border:1px solid #a7f3d0; color:#05603a; border-radius:9px; padding:9px; font-weight:750; }
.decision-hybrid { background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; border-radius:9px; padding:9px; font-weight:750; }
.decision-human { background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; border-radius:9px; padding:9px; font-weight:750; }
.history-item { border:1px solid #e4eaf1; border-radius:10px; padding:9px; margin-bottom:7px; background:#fff; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GROQ
# ============================================================
@st.cache_resource
def get_client():
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) if key else None

client = get_client()
if client is None:
    st.error("GROQ_API_KEY is not loaded. Load it in Colab before starting Streamlit.")
    st.stop()

# ============================================================
# DATABASE
# ============================================================
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        language TEXT NOT NULL DEFAULT 'English',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        message_id INTEGER NOT NULL,
        sentiment TEXT,
        emotion TEXT,
        intent TEXT,
        urgency TEXT,
        escalation_risk TEXT,
        key_issue TEXT,
        confidence INTEGER,
        decision TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        customer_message TEXT NOT NULL,
        agent_message TEXT NOT NULL,
        tone INTEGER,
        empathy INTEGER,
        clarity INTEGER,
        accuracy INTEGER,
        actionability INTEGER,
        coaching_tip TEXT,
        improved_response TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        ticket_code TEXT NOT NULL,
        department TEXT,
        summary TEXT,
        requested_action TEXT,
        urgency TEXT,
        escalation_risk TEXT,
        status TEXT NOT NULL DEFAULT 'Open',
        resolution TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );
    """)
    con.commit(); con.close()

init_db()

# ============================================================
# AUTH
# ============================================================
def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120000).hex()
    return f"{salt.hex()}:{digest}"

def verify_password(password, stored):
    try:
        salt_hex, digest = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        new_digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120000).hex()
        return hmac.compare_digest(new_digest, digest)
    except Exception:
        return False

def signup(name, email, password):
    con = db()
    try:
        con.execute("INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?)",
                     (name.strip(), email.strip().lower(), hash_password(password), datetime.now().isoformat(timespec="seconds")))
        con.commit(); return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    finally:
        con.close()

def signin(email, password):
    con = db(); row = con.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone(); con.close()
    return dict(row) if row and verify_password(password, row["password_hash"]) else None

# ============================================================
# CONVERSATIONS
# ============================================================
def create_conversation(user_id, title="New Conversation", language="English"):
    now = datetime.now().isoformat(timespec="seconds")
    con = db(); cur = con.execute(
        "INSERT INTO conversations(user_id,title,language,created_at,updated_at) VALUES(?,?,?,?,?)",
        (user_id, title, language, now, now)
    ); con.commit(); cid = cur.lastrowid; con.close(); return cid

def conversations(user_id):
    con = db(); rows = con.execute("SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC", (user_id,)).fetchall(); con.close(); return [dict(r) for r in rows]

def messages(cid):
    con = db(); rows = con.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (cid,)).fetchall(); con.close(); return [dict(r) for r in rows]

def add_message(cid, role, content):
    now = datetime.now().isoformat(timespec="seconds")
    con = db(); cur = con.execute("INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)", (cid,role,content,now))
    con.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now,cid)); con.commit(); mid = cur.lastrowid; con.close(); return mid

def save_analysis(cid, mid, a, decision):
    con = db(); con.execute("""INSERT INTO analyses(conversation_id,message_id,sentiment,emotion,intent,urgency,escalation_risk,key_issue,confidence,decision,created_at)
    VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (cid,mid,a.get("sentiment"),a.get("emotion"),a.get("intent"),a.get("urgency"),a.get("escalation_risk"),a.get("key_issue"),int(a.get("confidence",0)),decision,datetime.now().isoformat(timespec="seconds"))); con.commit(); con.close()

def save_evaluation(cid, customer, agent, f):
    con = db(); con.execute("""INSERT INTO evaluations(conversation_id,customer_message,agent_message,tone,empathy,clarity,accuracy,actionability,coaching_tip,improved_response,created_at)
    VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (cid,customer,agent,int(f.get("tone_score",0)),int(f.get("empathy_score",0)),int(f.get("clarity_score",0)),int(f.get("accuracy_score",0)),int(f.get("actionability_score",0)),f.get("coaching_tip",""),f.get("improved_response",""),datetime.now().isoformat(timespec="seconds"))); con.commit(); con.close()

def create_ticket(cid, handoff):
    code = f"TKT-{datetime.now().strftime('%m%d%H%M%S%f')[:-3]}"
    con = db(); con.execute("""INSERT INTO tickets(conversation_id,ticket_code,department,summary,requested_action,urgency,escalation_risk,created_at)
    VALUES(?,?,?,?,?,?,?,?)""", (cid,code,handoff.get("department","General Support"),handoff.get("summary",""),handoff.get("requested_action",""),handoff.get("urgency","medium"),handoff.get("escalation_risk","medium"),datetime.now().isoformat(timespec="seconds"))); con.commit(); con.close(); return code

def tickets_for_user(uid):
    con = db(); rows = con.execute("""SELECT t.* FROM tickets t JOIN conversations c ON t.conversation_id=c.id WHERE c.user_id=? ORDER BY CASE WHEN t.status='Open' THEN 0 ELSE 1 END, t.id DESC""", (uid,)).fetchall(); con.close(); return [dict(r) for r in rows]

def resolve_ticket(ticket_id, resolution):
    con = db(); con.execute("UPDATE tickets SET status='Resolved', resolution=? WHERE id=?", (resolution,ticket_id)); con.commit(); con.close()

# ============================================================
# HELPERS
# ============================================================
def detect_language(text):
    try:
        code = detect(text)
        return code, LANGUAGE_NAMES.get(code, code.upper())
    except Exception:
        return "en", "English"

def retrieve_kb(text, top_k=3):
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    ranked=[]
    for item in KB:
        overlap=0
        for key in item["keywords"]:
            kt=set(re.findall(r"[a-z0-9]+", key.lower()))
            if tokens & kt: overlap += 1
        if overlap: ranked.append((min(100,overlap*20),item))
    ranked.sort(key=lambda x:x[0], reverse=True)
    return ranked[:top_k]

def groq_json(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.15,
        response_format={"type":"json_object"},
    )
    return json.loads(response.choices[0].message.content or "{}")

def history_text(cid, limit=12):
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages(cid)[-limit:])

# ============================================================
# AI
# ============================================================
def analyze_customer(text, language, history):
    kb = retrieve_kb(text)
    kb_text = "\n".join(f"- {x[1]['category']}: {x[1]['answer']}" for x in kb) or "- No direct FAQ match."
    prompt = f"""
You are the real-time decision engine of an AI Customer Support Assistant.
Analyze the latest customer message using the conversation history.

Language: {language}
Conversation history:
{history}

Latest customer message:
{text}

Relevant knowledge:
{kb_text}

Return ONLY JSON:
{{
  "sentiment":"positive|neutral|negative",
  "emotion":"short emotion label",
  "intent":"Payment / Billing|Order & Delivery|Product Inquiry|Technical Issue|Complaint|Feedback|Refund / Cancellation|General Query",
  "urgency":"low|medium|high",
  "escalation_risk":"low|medium|high",
  "key_issue":"short specific issue",
  "confidence":0,
  "requires_human":false,
  "reason":"short reason"
}}
Rules: confidence is 0-100. Explicit refund/cancellation, money deducted/failed payment, strong frustration, high urgency, or low confidence may require human review. Do not invent customer facts.
"""
    return groq_json(prompt)

def decision_engine(text, a, kb):
    low = text.lower()
    special = any(x in low for x in ["refund","cancel","cancellation","money deducted","money back"])
    high = a.get("urgency","low").lower()=="high" or a.get("escalation_risk","low").lower()=="high"
    conf = int(a.get("confidence",0)); kb_score = kb[0][0] if kb else 0
    if special or high:
        return "Hybrid Assist" if conf >= 75 and kb_score >= 40 else "Human Escalation"
    return "AI Handle" if conf >= 75 and kb_score >= 40 else "Hybrid Assist"

def generate_reply(text, a, kb, language, history):
    kb_answer = kb[0][1]["answer"] if kb else "No directly matched FAQ is available."
    prompt = f"""
You are an expert customer-support assistant working alongside a human agent.
Write a concise customer-facing response in {language}.

Conversation history:
{history}

Latest customer message:
{text}

Customer analysis: {json.dumps(a)}
Knowledge-base guidance: {kb_answer}

Requirements: acknowledge concern, show empathy, explain next step when possible, never invent order/payment/account facts, never promise unsupported actions or timelines, and mention human review when required. Return only the response text.
"""
    r = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.25)
    return (r.choices[0].message.content or "").strip()

def evaluate_agent(customer, agent, language, history):
    prompt = f"""
You are an AI customer-support coach.

Conversation history:
{history}

Customer:
{customer}

Agent:
{agent}

Return ONLY JSON:
{{
  "tone_score":0,
  "empathy_score":0,
  "clarity_score":0,
  "accuracy_score":0,
  "actionability_score":0,
  "coaching_tip":"one concrete coaching tip",
  "improved_response":"improved response in {language}"
}}
All scores are 1-10. Do not invent information.
"""
    return groq_json(prompt)

def handoff_briefing(cid, text, a):
    prompt = f"""
Create a concise human-agent handoff briefing from this support conversation.
Conversation:
{history_text(cid)}
Latest customer message:
{text}
Analysis:
{json.dumps(a)}
Return ONLY JSON:
{{
 "summary":"2-3 sentence summary",
 "requested_action":"specific action for human team",
 "department":"Accounts / Refunds|Delivery|Technical Support|Product|General Support",
 "urgency":"{a.get('urgency','medium')}",
 "escalation_risk":"{a.get('escalation_risk','medium')}"
}}
"""
    return groq_json(prompt)

# ============================================================
# VOICE
# ============================================================
def transcribe(audio_file):
    r = client.audio.transcriptions.create(file=(audio_file.name,audio_file.getvalue()), model=WHISPER_MODEL, response_format="json", temperature=0.0)
    return r.text

@st.cache_data(ttl=3600)
def tts_voices():
    try: return asyncio.run(edge_tts.list_voices())
    except Exception: return []

def voice_for(code):
    voices = tts_voices()
    for v in voices:
        if str(v.get("Locale","")).lower().startswith(code.lower()):
            return v["ShortName"]
    return "en-US-JennyNeural"

def make_audio(text, code):
    out=io.BytesIO(); voice=voice_for(code)
    async def run():
        c=edge_tts.Communicate(text=text, voice=voice)
        async for chunk in c.stream():
            if chunk["type"]=="audio": out.write(chunk["data"])
    asyncio.run(run()); out.seek(0); return out

# ============================================================
# AUTH SCREEN
# ============================================================
if "user" not in st.session_state:
    st.session_state.user = None
if "cid" not in st.session_state:
    st.session_state.cid = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "reply" not in st.session_state:
    st.session_state.reply = None
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "decision" not in st.session_state:
    st.session_state.decision = None
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

if st.session_state.user is None:
    st.markdown(f"""
    <div class="topbar"><div class="brand">🤝 {APP_NAME}</div><div class="tagline">AI + Human customer support with real-time coaching, multilingual interaction and smart escalation</div></div>
    """, unsafe_allow_html=True)
    login, register = st.tabs(["Sign In", "Create Account"])
    with login:
        st.subheader("Welcome back")
        email=st.text_input("Email", key="login_email")
        pwd=st.text_input("Password", type="password", key="login_pwd")
        if st.button("Sign In", type="primary", use_container_width=True):
            u=signin(email,pwd)
            if u:
                st.session_state.user=u
                cs=conversations(u["id"])
                st.session_state.cid=cs[0]["id"] if cs else create_conversation(u["id"])
                st.rerun()
            else: st.error("Invalid email or password.")
    with register:
        st.subheader("Create your account")
        name=st.text_input("Full name", key="signup_name")
        email=st.text_input("Email address", key="signup_email")
        pwd=st.text_input("Password", type="password", key="signup_pwd")
        confirm=st.text_input("Confirm password", type="password", key="signup_confirm")
        if st.button("Create Account", type="primary", use_container_width=True):
            if not name or not email or not pwd: st.warning("Complete all fields.")
            elif pwd != confirm: st.error("Passwords do not match.")
            elif len(pwd)<6: st.error("Password must contain at least 6 characters.")
            else:
                ok,msg=signup(name,email,pwd)
                st.success(msg) if ok else st.error(msg)
    st.stop()

# ============================================================
# MAIN HEADER / SIDEBAR
# ============================================================
user=st.session_state.user
st.markdown(f"""
<div class="topbar"><div class="brand">🤝 {APP_NAME}</div><div class="tagline">Welcome, {escape(user['name'])} · Live customer support copilot</div></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Workspace")
    page=st.radio("Navigation", ["Support Workspace","Conversation History","Agent Evaluation","Handoff Queue","Analytics","Knowledge Base","Voice Assistant"], label_visibility="collapsed")
    st.markdown("---")
    lang=st.selectbox("Response Language", list(LANGUAGES.keys()))
    st.markdown("---")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.cid=create_conversation(user["id"], language="English")
        st.session_state.analysis=None; st.session_state.reply=None; st.session_state.feedback=None; st.session_state.decision=None; st.rerun()
    if st.button("Sign Out", use_container_width=True):
        st.session_state.user=None; st.session_state.cid=None; st.rerun()

cs=conversations(user["id"])
if not cs:
    st.session_state.cid=create_conversation(user["id"])
    cs=conversations(user["id"])
if st.session_state.cid not in [c["id"] for c in cs]: st.session_state.cid=cs[0]["id"]

open_count=sum(t["status"]=="Open" for t in tickets_for_user(user["id"]))
msg_count=sum(len(messages(c["id"])) for c in cs)
k1,k2,k3=st.columns(3)
for col,label,val in [(k1,"Conversations",len(cs)),(k2,"Messages",msg_count),(k3,"Open Handoffs",open_count)]:
    with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

# ============================================================
# SUPPORT WORKSPACE
# ============================================================
if page=="Support Workspace":
    left,center,right=st.columns([.92,1.65,.95])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Conversation History</div>', unsafe_allow_html=True)
        for c in cs[:15]:
            st.markdown(f'<div class="history-item"><b>{escape(c["title"])}</b><br><small>{escape(c["language"])} · {escape(c["updated_at"])}</small></div>', unsafe_allow_html=True)
            if st.button("Open", key=f"open_{c['id']}", use_container_width=True):
                st.session_state.cid=c["id"]; st.session_state.analysis=None; st.session_state.reply=None; st.session_state.feedback=None; st.session_state.decision=None; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with center:
        st.markdown('<div class="panel"><div class="panel-title">Live Conversation</div>', unsafe_allow_html=True)
        for m in messages(st.session_state.cid):
            cls="customer" if m["role"]=="customer" else "agent"
            icon="👤" if cls=="customer" else "👨‍💼"
            st.markdown(f'<div class="{cls}"><b>{icon} {m["role"].title()}</b> <small>{m["created_at"]}</small><br>{escape(m["content"])}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="panel-title">Customer Message</div>', unsafe_allow_html=True)
        customer=st.text_area("Message", value=st.session_state.voice_text, height=125, placeholder="Type the customer's message...", label_visibility="collapsed")
        audio=st.audio_input("Record customer message")
        a,b=st.columns(2)
        with a: trans=st.button("🎙 Voice → Text", use_container_width=True)
        with b: analyze=st.button("Analyze Customer", type="primary", use_container_width=True)
        if trans and audio:
            with st.spinner("Transcribing with Groq Whisper..."):
                try: st.session_state.voice_text=transcribe(audio); st.rerun()
                except Exception as e: st.error(f"Transcription error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        if analyze:
            if not customer.strip(): st.warning("Enter or record a customer message.")
            else:
                try:
                    if lang=="Auto Detect": code,language_name=detect_language(customer)
                    else: code,language_name=LANGUAGES[lang],lang
                    hist=history_text(st.session_state.cid)
                    kb=retrieve_kb(customer)
                    analysis=analyze_customer(customer,language_name,hist)
                    analysis["language"]=language_name; analysis["language_code"]=code
                    decision=decision_engine(customer,analysis,kb)
                    reply=generate_reply(customer,analysis,kb,language_name,hist)
                    mid=add_message(st.session_state.cid,"customer",customer)
                    con=db(); con.execute("UPDATE conversations SET title=?, language=? WHERE id=?", (analysis.get("key_issue","Customer Conversation")[:55],language_name,st.session_state.cid)); con.commit(); con.close()
                    save_analysis(st.session_state.cid,mid,analysis,decision)
                    st.session_state.analysis=analysis; st.session_state.decision=decision; st.session_state.reply=reply; st.session_state.feedback=None; st.session_state.voice_text=""
                    if decision in {"Human Escalation","Hybrid Assist"}:
                        handoff=handoff_briefing(st.session_state.cid,customer,analysis); create_ticket(st.session_state.cid,handoff)
                    st.rerun()
                except Exception as e: st.error(f"Analysis error: {e}")
        if st.session_state.reply:
            st.markdown('<div class="panel"><div class="panel-title">AI Suggested Response</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="reply">{escape(st.session_state.reply)}</div>', unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1:
                if st.button("Use This Reply", use_container_width=True):
                    add_message(st.session_state.cid,"agent",st.session_state.reply); st.rerun()
            with c2:
                if st.button("🔊 Play Reply", use_container_width=True):
                    code=st.session_state.analysis.get("language_code","en") if st.session_state.analysis else "en"
                    try: st.audio(make_audio(st.session_state.reply,code), format="audio/mp3")
                    except Exception as e: st.error(f"Voice output error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Live Intelligence</div>', unsafe_allow_html=True)
        a=st.session_state.analysis
        if a:
            for label,key in [("Language","language"),("Intent","intent"),("Sentiment","sentiment"),("Urgency","urgency"),("Escalation Risk","escalation_risk")]: st.metric(label,str(a.get(key,"Unknown")).title())
            st.metric("AI Confidence",f"{a.get('confidence',0)}%")
            st.write(f"**Emotion:** {a.get('emotion','')}")
            st.write(f"**Key Issue:** {a.get('key_issue','')}")
            st.write(f"**Reason:** {a.get('reason','')}")
            if st.session_state.decision=="AI Handle": st.markdown('<div class="decision-ai">AI Handle · Knowledge assisted</div>', unsafe_allow_html=True)
            elif st.session_state.decision=="Hybrid Assist": st.markdown('<div class="decision-hybrid">Hybrid Assist · Human review</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="decision-human">Human Escalation · Specialist team</div>', unsafe_allow_html=True)
        else: st.info("Run customer analysis to activate live intelligence.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="panel-title">Knowledge Base</div>', unsafe_allow_html=True)
        for score,item in retrieve_kb(customer if 'customer' in locals() else "")[:3]:
            st.markdown(f'<div class="history-item"><b>{escape(item["category"])}</b><br>{escape(item["title"])}<br><small>Match {score}%</small></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# HISTORY
# ============================================================
elif page=="Conversation History":
    st.markdown('<div class="panel"><div class="panel-title">Your Conversations</div>', unsafe_allow_html=True)
    for c in cs:
        st.markdown(f"### {c['title']}")
        st.caption(f"Language: {c['language']} · Updated: {c['updated_at']}")
        with st.expander(f"Conversation #{c['id']}"):
            for m in messages(c['id']): st.write(f"**{m['role'].title()}:** {m['content']}")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# AGENT EVALUATION
# ============================================================
elif page=="Agent Evaluation":
    st.markdown('<div class="panel"><div class="panel-title">Evaluate Agent Response</div>', unsafe_allow_html=True)
    ms=messages(st.session_state.cid); customers=[m for m in ms if m["role"]=="customer"]
    latest=customers[-1]["content"] if customers else ""
    if not latest: st.info("Analyze a customer message first.")
    else:
        st.write(f"**Customer:** {latest}")
        agent=st.text_area("Agent Response",height=150,placeholder="Enter the human agent's response...")
        if st.button("Evaluate Response",type="primary",use_container_width=True):
            if not agent.strip(): st.warning("Enter an agent response.")
            else:
                try:
                    language=st.session_state.analysis.get("language","English") if st.session_state.analysis else "English"
                    f=evaluate_agent(latest,agent,language,history_text(st.session_state.cid)); save_evaluation(st.session_state.cid,latest,agent,f); add_message(st.session_state.cid,"agent",agent); st.session_state.feedback=f; st.rerun()
                except Exception as e: st.error(f"Evaluation error: {e}")
        if st.session_state.feedback:
            f=st.session_state.feedback; cols=st.columns(5)
            for col,label,key in zip(cols,["Tone","Empathy","Clarity","Accuracy","Actionability"],["tone_score","empathy_score","clarity_score","accuracy_score","actionability_score"]):
                with col: st.metric(label,f"{f.get(key,0)}/10")
            st.markdown(f'<div class="coach"><b>Coaching Tip</b><br>{escape(str(f.get("coaching_tip","")))}<br><br><b>Improved Response</b><br>{escape(str(f.get("improved_response","")))}</div>',unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# HANDOFF
# ============================================================
elif page=="Handoff Queue":
    st.markdown('<div class="panel"><div class="panel-title">Human Handoff Queue</div>', unsafe_allow_html=True)
    ts=tickets_for_user(user["id"])
    if not ts: st.info("No handoff tickets yet.")
    else:
        selected=st.selectbox("Tickets",[f"{t['ticket_code']} · {t['urgency'].upper()} · {t['department']} · {t['status']}" for t in ts])
        code=selected.split("·")[0].strip(); t=next(x for x in ts if x["ticket_code"]==code)
        st.markdown(f'<div class="handoff"><b>Ticket:</b> {escape(t["ticket_code"])}<br><b>Department:</b> {escape(t["department"] or "General Support")}<br><b>Status:</b> {escape(t["status"])}<br><br><b>Summary</b><br>{escape(t["summary"] or "")}<br><br><b>Requested Action</b><br>{escape(t["requested_action"] or "")}</div>',unsafe_allow_html=True)
        x,y,z=st.columns(3)
        with x: st.metric("Urgency",str(t["urgency"] or "unknown").title())
        with y: st.metric("Escalation",str(t["escalation_risk"] or "unknown").title())
        with z: st.metric("Department",t["department"] or "General")
        if t["status"]=="Open":
            r1,r2,r3=st.columns(3)
            for col,label,value in [(r1,"Resolved by AI","AI"),(r2,"Resolved by Human","Human"),(r3,"Resolved by Hybrid","Hybrid")]:
                with col:
                    if st.button(label,use_container_width=True): resolve_ticket(t["id"],value); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# ANALYTICS
# ============================================================
elif page=="Analytics":
    st.markdown('<div class="panel"><div class="panel-title">Performance Analytics</div>', unsafe_allow_html=True)
    ts=tickets_for_user(user["id"]); resolved=[t for t in ts if t["status"]=="Resolved"]
    ai=sum(t["resolution"]=="AI" for t in resolved); human=sum(t["resolution"]=="Human" for t in resolved); hybrid=sum(t["resolution"]=="Hybrid" for t in resolved)
    a,b,c,d=st.columns(4)
    for col,label,val in [(a,"AI Resolved",ai),(b,"Human Resolved",human),(c,"Hybrid Resolved",hybrid),(d,"Total Handoffs",len(ts))]:
        with col: st.metric(label,val)
    df=pd.DataFrame({"Mode":["AI","Human","Hybrid"],"Cases":[ai,human,hybrid]}); st.bar_chart(df.set_index("Mode"))
    con=db(); rows=con.execute("""SELECT a.created_at,a.sentiment,a.emotion,a.intent,a.urgency,a.escalation_risk,a.key_issue,a.confidence,a.decision FROM analyses a JOIN conversations c ON a.conversation_id=c.id WHERE c.user_id=? ORDER BY a.id DESC""",(user["id"],)).fetchall(); con.close()
    if rows:
        adf=pd.DataFrame([dict(r) for r in rows]); st.dataframe(adf,use_container_width=True,hide_index=True)
        q1,q2=st.columns(2)
        with q1: st.download_button("Download CSV",adf.to_csv(index=False),"customer_support_analysis.csv","text/csv",use_container_width=True)
        with q2: st.download_button("Download JSON",adf.to_json(orient="records",indent=2),"customer_support_analysis.json","application/json",use_container_width=True)
    else: st.info("Run customer analyses to populate analytics.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# KNOWLEDGE BASE
# ============================================================
elif page=="Knowledge Base":
    st.markdown('<div class="panel"><div class="panel-title">Knowledge Base</div>', unsafe_allow_html=True)
    q=st.text_input("Search FAQs",placeholder="refund, payment, delivery, technical issue...")
    results=retrieve_kb(q,top_k=len(KB)) if q.strip() else [(100,x) for x in KB]
    for score,item in results:
        st.markdown(f'<div class="history-item"><b>{escape(item["category"])}</b><br><b>{escape(item["title"])}</b><br>{escape(item["answer"])}<br><small>Match {score}%</small></div>',unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# VOICE ASSISTANT
# ============================================================
elif page=="Voice Assistant":
    st.markdown('<div class="panel"><div class="panel-title">Voice Assistant</div>', unsafe_allow_html=True)
    st.write("Record customer speech → Groq speech-to-text → multilingual analysis → AI response → voice playback.")
    audio=st.audio_input("Record customer message")
    if st.button("🎙 Transcribe and Analyze",type="primary",use_container_width=True):
        if not audio: st.warning("Record a voice message first.")
        else:
            try:
                with st.spinner("Transcribing customer voice..."): text=transcribe(audio)
                st.session_state.voice_text=text
                if lang=="Auto Detect": code,language_name=detect_language(text)
                else: code,language_name=LANGUAGES[lang],lang
                kb=retrieve_kb(text); a=analyze_customer(text,language_name,history_text(st.session_state.cid)); a["language"]=language_name; a["language_code"]=code
                decision=decision_engine(text,a,kb); reply=generate_reply(text,a,kb,language_name,history_text(st.session_state.cid))
                mid=add_message(st.session_state.cid,"customer",text); save_analysis(st.session_state.cid,mid,a,decision)
                st.session_state.analysis=a; st.session_state.decision=decision; st.session_state.reply=reply
            except Exception as e: st.error(f"Voice processing error: {e}")
    if st.session_state.voice_text:
        st.subheader("Transcript"); st.info(st.session_state.voice_text)
    if st.session_state.reply:
        st.subheader("Multilingual AI Response")
        st.markdown(f'<div class="reply">{escape(st.session_state.reply)}</div>',unsafe_allow_html=True)
        if st.button("🔊 Play Voice Response",use_container_width=True):
            code=st.session_state.analysis.get("language_code","en") if st.session_state.analysis else "en"
            try: st.audio(make_audio(st.session_state.reply,code),format="audio/mp3")
            except Exception as e: st.error(f"Voice output error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("AI assists human agents; complex, specialized, high-risk or low-confidence cases can be routed to the appropriate human team.")
