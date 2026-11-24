import glob
import os
import sys
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(
    page_title="Ola Data Agentic AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. Load secrets (local .env + Streamlit Cloud st.secrets)
load_dotenv()
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass

# 2. Ensure root and Agents/ are on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(ROOT_DIR, "Agents")
for path in (ROOT_DIR, AGENTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# 3. Custom Midnight-Slate CSS (matches portfolio aesthetic)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0b1120;
        --panel: #0f172a;
        --line: rgba(148, 163, 184, 0.16);
        --text: #e6edf7;
        --muted: #8a99b0;
        --accent: #5eead4;
        --accent-soft: rgba(94, 234, 212, 0.10);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, [data-testid="stDeployButton"] { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--panel);
        border-right: 1px solid var(--line);
    }

    /* Main container width for natural reading */
    .block-container {
        max-width: 860px;
        padding-top: 2.2rem;
        padding-bottom: 5rem;
    }

    /* Custom hero header */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 12px;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0 0 6px 0;
        color: var(--text);
    }
    .hero-sub {
        color: var(--muted);
        font-size: 0.98rem;
        margin-bottom: 1.5rem;
    }

    /* Route tag pill */
    .route-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        background: var(--accent-soft);
        border: 1px solid rgba(94, 234, 212, 0.25);
        color: var(--accent);
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        margin-bottom: 8px;
    }

    /* Buttons */
    .stButton > button {
        background-color: rgba(15, 23, 42, 0.85);
        border: 1px solid var(--line);
        color: var(--text);
        border-radius: 10px;
        text-align: left;
        transition: border-color 0.2s, background-color 0.2s;
    }
    .stButton > button:hover {
        border-color: var(--accent);
        background-color: var(--accent-soft);
        color: var(--accent);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: rgba(15, 23, 42, 0.55);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_agents():
    from Agents.data_agent import main_agent, content_to_text, sql_analyst, etl_analyst

    return main_agent, content_to_text, sql_analyst, etl_analyst


def find_latest_etl_file(since_ts: float):
    """Find the most recent CSV/JSON written during an ETL run so users can preview & download it."""
    folders = ["data/extract", "data/output", "data/tranform", "data/transform"]
    candidates = []
    for folder in folders:
        full_dir = os.path.join(ROOT_DIR, folder)
        if os.path.isdir(full_dir):
            for ext in ("*.csv", "*.json"):
                for fp in glob.glob(os.path.join(full_dir, ext)):
                    mtime = os.path.getmtime(fp)
                    if mtime >= since_ts - 1.0:
                        candidates.append((mtime, fp))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def friendly_error(err: Exception) -> str:
    msg = str(err)
    if "503" in msg or "UNAVAILABLE" in msg:
        return "The free-tier Gemini model is briefly busy right now. Wait 2–3 seconds and send your question again."
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "Free-tier API rate limit reached for a moment. Please wait a few seconds and try again."
    return f"Something went wrong while running the agent:\n\n`{msg}`"

def build_lc_history(messages: list, limit: int = 8) -> list:
    """Convert recent Streamlit session messages into LangChain message objects."""
    lc_msgs = []
    for m in messages[-limit:]:
        if m["role"] == "user":
            lc_msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_msgs.append(AIMessage(content=m["content"]))
    return lc_msgs

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---- Left Sidebar ----
with st.sidebar:
    st.markdown("### Ola Data Agentic AI")
    st.caption("Built by Jeevan Parihar • LangGraph + Neon PostgreSQL")

    with st.expander("What is this & what can it do?", expanded=True):
        st.markdown(
            """
            A **multi-agent data assistant** powered by a LangGraph supervisor. Just ask in plain English:

            * **SQL Analyst** — writes safe, read-only PostgreSQL queries over **61,000+ Ola records** and explains the answer.
            * **ETL Analyst** — fetches live data from public APIs and transforms/exports CSV files with Pandas.
            * **General Chat** — ask questions about how the system works.
            """
        )

    with st.expander("Database Tables (5)"):
        st.markdown(
            """
            * `users` — 10,000 riders & drivers (city, province, signup)
            * `vehicles` — 3,000 cars (make, model, year, color)
            * `rides` — 20,000 trips (distance_km, fare, surge, status)
            * `payments` — 16,073 transactions (method, status, amount)
            * `ratings` — 12,000 reviews (1–5 stars, comments)
            """
        )

    st.divider()
    agent_mode = st.radio(
        "Agent Mode",
        ["Auto (Supervisor)", "SQL Analyst", "ETL Pipeline"],
        help="Auto lets the supervisor route your message. Pick SQL or ETL directly to skip the router and answer faster.",
    )
    show_sql = st.toggle("Show generated SQL", value=True)
    show_timing = st.toggle("Show response time", value=True)

    st.divider()
    if st.button("New chat", use_container_width=True, icon=":material/refresh:"):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

# ---- Header ----
st.markdown(
    """
    <div class="hero-badge">● MULTI-AGENT DATA ASSISTANT</div>
    <h1 class="hero-title">Ola Data Agentic AI</h1>
    <p class="hero-sub">Query 61k+ cloud PostgreSQL records in plain English or run an automated API-to-CSV ETL pipeline.</p>
    """,
    unsafe_allow_html=True,
)

# ---- Empty-State Starter Cards (only shown before first message) ----
STARTERS = [
    ("SQL • Payments", "What are the different payment methods in our database?"),
    ("SQL • Cities", "Show the top 5 cities with the highest number of users."),
    ("SQL • Rides", "What is the average ride fare and distance for completed rides?"),
    ("ETL • Live API", "Extract data from https://pokeapi.co/api/v2/pokemon and save it as a csv in data/extract"),
]

if not st.session_state.messages:
    st.caption("Try asking:")
    col1, col2 = st.columns(2)
    for idx, (label, text) in enumerate(STARTERS):
        with (col1 if idx % 2 == 0 else col2):
            if st.button(f"**{label}**\n\n{text}", key=f"starter_{idx}", use_container_width=True):
                st.session_state.pending_prompt = text
                st.rerun()

# ---- Render Chat History ----
for i, msg in enumerate(st.session_state.messages):
    avatar = ":material/person:" if msg["role"] == "user" else ":material/auto_awesome:"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant" and msg.get("route"):
            t_str = f" · {msg['elapsed']:.2f}s" if (show_timing and msg.get("elapsed") is not None) else ""
            st.markdown(f'<span class="route-pill">{msg["route"]}{t_str}</span>', unsafe_allow_html=True)

        st.markdown(msg["content"])

        if show_sql and msg.get("sql_query"):
            with st.expander("View generated SQL"):
                st.code(msg["sql_query"], language="sql")

        if msg.get("etl_file") and os.path.exists(msg["etl_file"]):
            fp = msg["etl_file"]
            with st.expander(f"Preview & Download: {os.path.basename(fp)}"):
                if fp.endswith(".csv"):
                    try:
                        st.dataframe(pd.read_csv(fp).head(15), use_container_width=True)
                    except Exception:
                        pass
                with open(fp, "rb") as f:
                    st.download_button(
                        label=f"Download {os.path.basename(fp)}",
                        data=f.read(),
                        file_name=os.path.basename(fp),
                        key=f"dl_{i}",
                    )

# ---- Handle New Input ----
chat_prompt = st.chat_input("Ask about Ola rides, drivers, revenue, or run an ETL task...")
prompt = chat_prompt or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        with st.spinner("Thinking..."):
            start_perf = time.perf_counter()
            start_wall = time.time()
            route_label, sql_query, etl_file = "", "", None

            try:
                main_agent, content_to_text, sql_analyst, etl_analyst = load_agents()
                lc_history = build_lc_history(st.session_state.messages, limit=8)

                if agent_mode == "SQL Analyst":
                    res = sql_analyst.invoke({"user_question": prompt})
                    route_label = "SQL Analyst"
                    answer = content_to_text(res["messages"][-1].content)
                    sql_query = res.get("generated_sql_query", "")
                elif agent_mode == "ETL Pipeline":
                    res = etl_analyst.invoke({"messages": lc_history})
                    route_label = "ETL Analyst"
                    answer = content_to_text(res["messages"][-1].content)
                    etl_file = find_latest_etl_file(start_wall)
                else:
                    res = main_agent.invoke({"messages": lc_history})
                    raw_route = res.get("route_response", "").strip().lower()
                    route_map = {"sql": "SQL Analyst", "etl": "ETL Analyst", "chat": "Assistant"}
                    route_label = route_map.get(raw_route, "Assistant")
                    final_msg = res["messages"][-1]
                    answer = content_to_text(final_msg.content)
                    sql_query = getattr(final_msg, "additional_kwargs", {}).get("sql_query", "")
                    if raw_route == "etl":
                        etl_file = find_latest_etl_file(start_wall)
            except Exception as e:
                answer = friendly_error(e)

            elapsed = time.perf_counter() - start_perf

        if route_label:
            t_str = f" · {elapsed:.2f}s" if show_timing else ""
            st.markdown(f'<span class="route-pill">{route_label}{t_str}</span>', unsafe_allow_html=True)

        st.markdown(answer)

        if show_sql and sql_query:
            with st.expander("View generated SQL"):
                st.code(sql_query, language="sql")

        if etl_file and os.path.exists(etl_file):
            with st.expander(f"Preview & Download: {os.path.basename(etl_file)}"):
                if etl_file.endswith(".csv"):
                    try:
                        st.dataframe(pd.read_csv(etl_file).head(15), use_container_width=True)
                    except Exception:
                        pass
                with open(etl_file, "rb") as f:
                    st.download_button(
                        label=f"Download {os.path.basename(etl_file)}",
                        data=f.read(),
                        file_name=os.path.basename(etl_file),
                        key=f"dl_live_{int(start_wall)}",
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "route": route_label,
            "elapsed": elapsed,
            "sql_query": sql_query,
            "etl_file": etl_file,
        }
    )