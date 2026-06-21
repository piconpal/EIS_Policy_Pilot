"""
chat_app.py — Streamlit chatbot UI for Enterprise RAG
Talks to the FastAPI backend running on localhost:8000.

Run:
    streamlit run ui/chat_app.py
"""

import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise RAG — Security Analytics",
    page_icon="🔐",
    layout="wide",
)

# ── Session state init ─────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Helpers ────────────────────────────────────────────────────────────────────

def _health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _query(query: str, search_mode: str | None) -> dict:
    payload = {
        "query":      query,
        "session_id": st.session_state.session_id,
        "user_id":    st.session_state.session_id,
    }
    if search_mode:
        payload["search_mode"] = search_mode
    r = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _get_logs(limit: int = 100) -> dict:
    try:
        r = requests.get(f"{API_BASE}/logs", params={"limit": limit}, timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔐 Enterprise RAG")
    st.caption("Security Analytics Assistant")

    st.divider()

    health = _health()
    if health:
        st.success("API Online")
        st.caption(f"Model: `{health['model']}`")
        st.caption(f"Default mode: `{health['search_mode']}`")
    else:
        st.error("API Offline — start the FastAPI server first")
        st.code("python -m uvicorn src.api.app:app --port 8000")

    st.divider()

    search_mode = st.selectbox(
        "Search mode",
        options=["(use default)", "hybrid", "vector", "bm25"],
        index=0,
        help="Override the search strategy for this session.",
    )
    effective_mode = None if search_mode == "(use default)" else search_mode

    st.divider()

    st.caption(f"Session ID: `{st.session_state.session_id}`")
    st.caption(f"Turns: {len([m for m in st.session_state.messages if m['role'] == 'user'])}")

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()

    st.markdown(f"[📖 API Docs]({API_BASE}/docs)  •  [📊 Eval JSON]({API_BASE}/eval)")


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_chat, tab_analytics = st.tabs(["💬 Chat", "📊 Analytics"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.header("Security Analytics Assistant", divider="gray")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                if meta.get("sources_cited"):
                    with st.expander(f"📄 {len(meta['sources_cited'])} source(s) cited"):
                        for src in meta["sources_cited"]:
                            header = f" — {src['section_header']}" if src.get("section_header") else ""
                            st.markdown(f"- **{src['source_file']}** p{src['page_number']}{header}")
                cols = st.columns(4)
                cols[0].caption(f"⏱ {meta.get('latency_ms', 0)/1000:.1f}s")
                cols[1].caption(f"🔍 {meta.get('query_type', '—')}")
                cols[2].caption(f"📥 {meta.get('prompt_tokens', 0)} tok")
                cols[3].caption(f"📤 {meta.get('completion_tokens', 0)} tok")

    if prompt := st.chat_input("Ask a security question…", disabled=(health is None)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                try:
                    result = _query(prompt, effective_mode)

                    if not result["is_safe"]:
                        st.warning(f"⚠️ {result['answer']}")
                        st.session_state.messages.append({
                            "role": "assistant", "content": f"⚠️ {result['answer']}", "meta": None,
                        })
                    else:
                        answer = result["answer"]
                        st.markdown(answer)

                        if result.get("sources_cited"):
                            with st.expander(f"📄 {len(result['sources_cited'])} source(s) cited"):
                                for src in result["sources_cited"]:
                                    header = f" — {src['section_header']}" if src.get("section_header") else ""
                                    st.markdown(f"- **{src['source_file']}** p{src['page_number']}{header}")

                        cols = st.columns(4)
                        cols[0].caption(f"⏱ {result.get('latency_ms', 0)/1000:.1f}s")
                        cols[1].caption(f"🔍 {result.get('query_type', '—')}")
                        cols[2].caption(f"📥 {result.get('prompt_tokens', 0)} tok")
                        cols[3].caption(f"📤 {result.get('completion_tokens', 0)} tok")

                        st.session_state.messages.append({
                            "role": "assistant", "content": answer, "meta": result,
                        })

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API. Is the FastAPI server running on port 8000?")
                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

with tab_analytics:
    st.header("Retrieval Logger — Analytics", divider="gray")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    data = _get_logs(limit=200)
    stats = data.get("stats", {})
    logs  = data.get("logs",  [])

    if not stats:
        st.warning("No data yet — make some queries in the Chat tab first.")
        st.stop()

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.subheader("Production Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Queries",    stats.get("total_queries", 0))
    c2.metric("Safe Queries",     stats.get("safe_queries",  0))
    c3.metric("Blocked Queries",  stats.get("blocked_queries", 0))
    c4.metric("Unique Sessions",  stats.get("unique_sessions", 0))
    safe_rate = (
        round(stats["safe_queries"] / stats["total_queries"] * 100, 1)
        if stats.get("total_queries") else 0
    )
    c5.metric("Safe Rate", f"{safe_rate}%")

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Latency breakdown ─────────────────────────────────────────────────────
    with col_left:
        st.subheader("Avg Latency Breakdown (ms)")
        latency_data = {
            "Stage": ["Retriever", "Reranker", "LLM"],
            "Avg ms": [
                stats.get("avg_retriever_ms", 0),
                stats.get("avg_reranker_ms",  0),
                stats.get("avg_llm_ms",        0),
            ],
        }
        st.bar_chart(
            data=dict(zip(latency_data["Stage"], latency_data["Avg ms"])),
            use_container_width=True,
        )
        st.caption(f"End-to-end avg: **{stats.get('avg_latency_ms', 0):.0f} ms**")

    # ── Top cited sources ─────────────────────────────────────────────────────
    with col_right:
        st.subheader("Top Cited Documents")
        top_sources = stats.get("top_sources", [])
        if top_sources:
            src_data = {s["source_file"]: s["cited_count"] for s in top_sources}
            st.bar_chart(src_data, use_container_width=True)
        else:
            st.info("No citations logged yet.")

    st.divider()

    # ── Token usage ───────────────────────────────────────────────────────────
    st.subheader("Token Usage")
    t1, t2, t3 = st.columns(3)
    t1.metric("Avg Prompt Tokens",     int(stats.get("avg_prompt_tokens",     0)))
    t2.metric("Avg Completion Tokens", int(stats.get("avg_completion_tokens", 0)))
    t3.metric("Avg Citations / Query", round(stats.get("avg_citations", 0), 1))

    st.divider()

    # ── Recent logs table ─────────────────────────────────────────────────────
    st.subheader("Recent Query Log")

    if logs:
        import pandas as pd

        rows = []
        for log in logs:
            rows.append({
                "Timestamp":    (log.get("timestamp") or "")[:19].replace("T", " "),
                "Query":        (log.get("query_text") or "")[:80],
                "Safe":         "✅" if log.get("input_safe") else "⛔",
                "Mode":         log.get("search_mode", "—"),
                "Query Type":   "—",
                "Latency (ms)": log.get("latency_ms") or 0,
                "Ret (ms)":     log.get("retriever_latency_ms") or 0,
                "LLM (ms)":     log.get("llm_latency_ms") or 0,
                "Prompt tok":   log.get("prompt_tokens") or 0,
                "Chunks ret":   log.get("chunks_retrieved") or 0,
                "Citations":    log.get("citation_count") or 0,
                "Session":      log.get("session_id", "—"),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("No logs yet.")
