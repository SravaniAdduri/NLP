"""Streamlit Frontend - Hallucination Detection & Fact Verification"""

import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="Hallucination Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- API CALLS ---

def api_health():
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=5).json()
    except Exception:
        return {"status": "offline", "index_size": 0}

def api_upload_file(file):
    try:
        files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
        r = requests.post(f"{API_BASE_URL}/upload/file", files=files, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": e.response.text}
    except Exception as e:
        return {"error": str(e)}

def api_reset_kb():
    try:
        r = requests.post(f"{API_BASE_URL}/upload/reset", timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": e.response.text}
    except Exception as e:
        return {"error": str(e)}

def api_simple_rag(query):
    try:
        r = requests.post(f"{API_BASE_URL}/simple_rag", json={"query": query, "llm_response": ""}, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": e.response.text}
    except Exception as e:
        return {"error": str(e)}

def api_multi_agent(query, llm_response=""):
    try:
        r = requests.post(f"{API_BASE_URL}/query", json={"query": query, "llm_response": llm_response}, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": e.response.text}
    except Exception as e:
        return {"error": str(e)}

def api_verify_response(text, evidence=None):
    try:
        payload = {"text": text, "evidence": evidence or []}
        r = requests.post(f"{API_BASE_URL}/verify_response", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": e.response.text}
    except Exception as e:
        return {"error": str(e)}

# --- DISPLAY HELPERS ---

def render_sentence_highlight(sentence_results):
    for r in sentence_results:
        label, conf, sent = r["label"], r["confidence"], r["sentence"]
        if label == "SUPPORTED":
            color, border, icon = "#d4edda", "#28a745", "✅"
        elif label == "CONTRADICTED":
            color, border, icon = "#f8d7da", "#dc3545", "❌"
        else:
            color, border, icon = "#fff3cd", "#ffc107", "⚠️"
        st.markdown(
            f'<div style="background:{color}; padding:10px; margin:6px 0; '
            f'border-radius:6px; border-left:5px solid {border};">'
            f'<b>{icon} {label}</b> (confidence: {conf:.2f})<br>{sent}</div>',
            unsafe_allow_html=True,
        )

def show_hallucination_methodology():
    """Display explanation of how hallucination is calculated"""
    with st.expander("📖 How is Hallucination Calculated?", expanded=False):
        st.markdown("""
### Hallucination Detection Methodology

The system uses **Natural Language Inference (NLI)** to detect hallucinations. Here's how it works:

#### Step 1: Response Segmentation
- The response is split into individual sentences
- Each sentence is analyzed independently

#### Step 2: Sentence-Level Classification (NLI)
For each sentence in the response, the system checks it against the **top-5 most relevant evidence passages** from your document:

| Label | Meaning | What it means |
|-------|---------|---------------|
| **✅ SUPPORTED** | The sentence is confirmed by the evidence | The statement is factually correct based on your document |
| **❌ CONTRADICTED** | The sentence contradicts the evidence | The statement conflicts with what's in your document (HALLUCINATION) |
| **⚠️ NEUTRAL / UNCERTAIN** | The evidence doesn't directly confirm or deny it | The statement cannot be verified from your document (potential hallucination) |

#### Step 3: Hallucination Rate Calculation
```
Hallucination Rate = (Contradicted + Uncertain) / Total Sentences
```

**Example:**
- Total sentences: 10
- Supported: 6 ✅
- Contradicted: 2 ❌
- Uncertain: 2 ⚠️
- **Hallucination Rate = (2 + 2) / 10 = 40%**

#### Step 4: Confidence Scoring
Each classification (SUPPORTED, CONTRADICTED, NEUTRAL) has a **confidence score** (0.0 to 1.0):
- **High confidence (0.8-1.0)**: The system is very certain about this classification
- **Medium confidence (0.5-0.8)**: Moderate certainty
- **Low confidence (0.0-0.5)**: Uncertain classification

**Lower confidence = less reliable hallucination detection** for that sentence

#### What Makes a Response Less Hallucinating?
- ✅ High percentage of SUPPORTED sentences
- ✅ Low percentage of CONTRADICTED sentences
- ✅ High confidence scores
- ✅ Factual statements grounded in the uploaded document

#### What Indicates More Hallucination?
- ❌ High CONTRADICTED rate (direct lies)
- ❌ High UNCERTAIN rate (unverifiable claims)
- ❌ Low confidence scores
- ❌ Statements that don't match the document content
        """)

def render_evidence(evidence_list, scores=None):
    if not evidence_list:
        st.info("No evidence retrieved.")
        return
    for i, ev in enumerate(evidence_list, 1):
        score_str = f" (relevance: {scores[i-1]:.2f})" if scores and i-1 < len(scores) else ""
        with st.expander(f"📖 Evidence [{i}]{score_str}"):
            st.write(ev)


def render_architecture_panel():
    """Render architecture diagrams for demo/report alignment."""
    with st.expander("🏗️ Architecture: Simple RAG vs Multi-Agent", expanded=False):
        tab_simple, tab_multi = st.tabs(["Simple RAG Pipeline", "Multi-Agent RAG Pipeline"])

        with tab_simple:
            simple_dot = """
            digraph SimpleRAG {
                rankdir=LR;
                node [shape=box, style="rounded,filled", fillcolor="#EAF3FF", color="#4A78C2",
                      fontname="Helvetica", fontsize=10, width=1.2, height=0.4];
                edge [color="#4A78C2"];
                nodesep=0.3; ranksep=0.4;

                q [label="Query"];
                emb [label="Embed"];
                vs [label="Vector Search"];
                ctx [label="Context"];
                gen [label="Generate"];
                ans [label="Answer"];

                q -> emb -> vs -> ctx -> gen -> ans;
            }
            """
            st.graphviz_chart(simple_dot, use_container_width=True)

        with tab_multi:
            multi_dot = """
            digraph MultiAgentRAG {
                rankdir=LR;
                node [shape=box, style="rounded,filled", fillcolor="#E9FFF1", color="#2E8B57",
                      fontname="Helvetica", fontsize=10, width=1.2, height=0.4];
                edge [color="#2E8B57"];
                nodesep=0.25; ranksep=0.35;

                q [label="Query"];
                plan [label="Planner"];

                qa [label="Rewrite"];
                ret [label="Retrieve"];
                meta [label="Metadata"];

                col [label="Collect"];
                rank [label="Rerank"];
                ver [label="Verify"];
                fact [label="Fact Check"];
                gen [label="Generate"];
                ans [label="Response"];

                q -> plan;
                plan -> qa;
                plan -> ret;
                plan -> meta;

                qa -> col;
                ret -> col;
                meta -> col;

                col -> rank -> ver -> fact -> gen -> ans;
            }
            """
            st.graphviz_chart(multi_dot, use_container_width=True)

# --- MAIN PAGE ---

def main():
    st.title("🔍 Hallucination Detection & Fact Verification")
    st.caption("Multi-Agent RAG Framework — Compare Simple RAG vs Multi-Agent Pipeline")
    render_architecture_panel()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    health = api_health()
    if health["status"] == "offline":
        st.error("❌ Backend API offline. Start with: `uvicorn api.main:app --reload --port 8000`")
        return

    # === SECTION 1: UPLOAD DOCUMENT ===
    st.subheader("📄 Upload Documents")
    col_upload, col_status = st.columns([3, 1])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Choose PDF, TXT, MD, or HTML files",
            type=["pdf", "txt", "md", "html"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
    with col_status:
        chunks = health.get("index_size", 0)
        if chunks > 0:
            st.success(f"✅ **{chunks}** chunks indexed")
        else:
            st.warning("No document yet")

    action_col1, action_col2, _ = st.columns([1, 1, 6])
    with action_col1:
        if st.button("🧹 Clear KB", key="btn_clear_kb", help="Clear all indexed documents"):
            result = api_reset_kb()
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Knowledge base cleared.")
                st.session_state.chat_history = []
                st.rerun()

    if uploaded_files:
        st.caption(f"📎 Selected {len(uploaded_files)} file(s)")
        with action_col2:
            if st.button("➤", key="btn_upload", type="primary", help="Upload and index selected documents"):
                total_chunks = 0
                failures = []
                for f in uploaded_files:
                    with st.spinner(f"Indexing {f.name}..."):
                        result = api_upload_file(f)
                    if "error" in result:
                        failures.append(f"{f.name}: {result['error']}")
                    else:
                        total_chunks += int(result.get("num_chunks", 0))

                if failures:
                    st.error("Some files failed to upload:")
                    for fail in failures:
                        st.write(f"- {fail}")
                if total_chunks > 0:
                    st.success(f"✅ Indexed **{total_chunks}** new chunks from {len(uploaded_files) - len(failures)} file(s).")
                    st.rerun()

    has_docs = health.get("index_size", 0) > 0
    if not has_docs:
        st.info("⬆️ Upload a document first, then ask questions below.")
        return

    st.markdown("---")

    # === SECTION 2: CONVERSATIONAL CHAT ===
    st.subheader("💬 Chat")
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.markdown("**Simple RAG Answer**")
            st.info(turn.get("simple_response", "No response."))
            st.markdown("**Multi-Agent Answer**")
            st.success(turn.get("agent_response", "No response."))

            if turn.get("rag_hall_rate") is not None and turn.get("agent_hall_rate") is not None:
                col_a, col_b = st.columns(2)
                col_a.metric("Simple Hallucination", f"{turn['rag_hall_rate']:.0%}")
                col_b.metric("Multi-Agent Hallucination", f"{turn['agent_hall_rate']:.0%}")

    query = st.chat_input("Ask anything about your uploaded documents...")

    if query:
        # Show the user's question immediately so it stays visible
        with st.chat_message("user"):
            st.write(query)

        st.markdown("---")
        col_rag, col_agent = st.columns(2)

        rag = None
        agent = None

        # Step 1: Run Simple RAG first (fast, naive)
        with col_rag:
            st.subheader("📄 Simple RAG")
            with st.spinner("Retrieving..."):
                rag = api_simple_rag(query)
            if "error" in rag:
                st.error(rag["error"])
            else:
                st.markdown("**Answer:**")
                st.info(rag.get("response", "No response."))
                st.caption(f"⏱️ {rag.get('latency_ms', 0):.0f} ms | Method: Direct vector search (no reranking)")
                render_evidence(rag.get("evidence", []), rag.get("evidence_scores", []))

        # Step 2: Run Multi-Agent directly from the query
        with col_agent:
            st.subheader("🤖 Multi-Agent Pipeline")
            with st.spinner("Running multi-agent pipeline..."):
                agent = api_multi_agent(query)
            if "error" in agent:
                st.error(agent["error"])
            else:
                st.markdown("**Corrected Answer:**")
                agent_response = agent.get("corrected_response") or "No response."
                st.success(agent_response)
                latency = agent.get("metrics", {}).get("latency_ms", 0) if agent.get("metrics") else 0
                st.caption(f"⏱️ {latency:.0f} ms | Method: Query rewrite → Multi-query retrieval → Rerank → Generate")
                render_evidence(agent.get("ranked_evidence", []))
                if agent.get("citations"):
                    with st.expander("📑 Citations"):
                        for c in agent["citations"]:
                            st.markdown(f"**[{c['id']}]** {c['text']}")

        # Automatic hallucination analysis for both generated responses
        rag_verification = None
        agent_verification = None
        agent_native_report = None
        with st.spinner("Running hallucination analysis..."):
            # Verify Simple RAG final response
            if rag and "error" not in rag and rag.get("response"):
                fair_evidence = rag.get("evidence", [])
                rag_verification = api_verify_response(
                    rag["response"],
                    fair_evidence,
                )

            # Verify Multi-Agent corrected final response
            if agent and "error" not in agent and agent.get("corrected_response"):
                # Prefer native in-pipeline hallucination report; fallback to standalone verifier.
                agent_native_report = agent.get("hallucination_report")
                if not (agent_native_report and agent_native_report.get("sentence_results")):
                    agent_evidence = agent.get("generated_evidence") or agent.get("ranked_evidence", [])
                    agent_verification = api_verify_response(
                        agent.get("corrected_response", ""),
                        agent_evidence,
                    )

        st.markdown("---")
        st.subheader("🔬 Hallucination Analysis (Auto)")
        st.caption("Fair comparison: both outputs are scored with the same verifier against the same evidence set.")

        col_h1, col_h2 = st.columns(2)

        rag_hall_rate = 0.0
        rag_available = False
        with col_h1:
            st.subheader("📄 Simple RAG")
            if rag_verification and "error" not in rag_verification and rag_verification.get("sentence_results"):
                rag_available = True
                rv = rag_verification
                rag_hall_rate = rv.get("hallucination_rate", 0.0)
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Supported", rv.get("supported_count", 0))
                c2.metric("❌ Contradicted", rv.get("contradicted_count", 0))
                c3.metric("⚠️ Uncertain", rv.get("neutral_count", 0))
                st.metric("Hallucination Rate", f"{rag_hall_rate:.0%}")
                render_sentence_highlight(rv.get("sentence_results", []))
            else:
                st.info("Could not analyze Simple RAG response.")

        agent_hall_rate = 0.0
        agent_available = False
        with col_h2:
            st.subheader("🤖 Multi-Agent Corrected Response")
            if agent_native_report and agent_native_report.get("sentence_results"):
                agent_available = True
                av = agent_native_report
                agent_hall_rate = av.get("hallucination_rate", 0.0)
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Supported", av.get("supported_count", 0))
                c2.metric("❌ Contradicted", av.get("contradicted_count", 0))
                c3.metric("⚠️ Uncertain", av.get("neutral_count", 0))
                st.metric("Hallucination Rate", f"{agent_hall_rate:.0%}")
                render_sentence_highlight(av.get("sentence_results", []))
                st.caption("Source: Multi-agent in-pipeline hallucination detection")
            elif agent_verification and "error" not in agent_verification and agent_verification.get("sentence_results"):
                agent_available = True
                av = agent_verification
                agent_hall_rate = av.get("hallucination_rate", 0.0)
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Supported", av.get("supported_count", 0))
                c2.metric("❌ Contradicted", av.get("contradicted_count", 0))
                c3.metric("⚠️ Uncertain", av.get("neutral_count", 0))
                st.metric("Hallucination Rate", f"{agent_hall_rate:.0%}")
                render_sentence_highlight(av.get("sentence_results", []))
                st.caption("Source: Standalone verifier fallback")
            else:
                st.info("Could not analyze Multi-Agent corrected response.")

        st.markdown("---")
        st.subheader("🏁 Which Hallucinates More?")
        if rag_available and agent_available:
            if rag_hall_rate > agent_hall_rate:
                diff = rag_hall_rate - agent_hall_rate
                st.error(
                    f"**Simple RAG hallucinates more:** {rag_hall_rate:.0%} hallucination rate.\n\n"
                    f"The Multi-Agent pipeline detected and corrected issues, reducing "
                    f"hallucination by **{diff:.0%}** (from {rag_hall_rate:.0%} → {agent_hall_rate:.0%})."
                )
            elif agent_hall_rate > rag_hall_rate:
                diff = agent_hall_rate - rag_hall_rate
                st.warning(
                    f"**Multi-Agent shows higher hallucination rate:** {agent_hall_rate:.0%} vs "
                    f"Simple RAG at {rag_hall_rate:.0%}.\n\n"
                    f"Note: This is the hallucination rate of the *original* Simple RAG response "
                    f"as analyzed by the Multi-Agent pipeline's NLI model with reranked evidence."
                )
            else:
                st.success(f"Both analyses agree at **{rag_hall_rate:.0%}** hallucination rate.")

            c1, c2 = st.columns(2)
            c1.metric("Simple RAG Hallucination", f"{rag_hall_rate:.0%}")
            c2.metric("Multi-Agent Detection", f"{agent_hall_rate:.0%}")
        elif rag_available:
            st.metric("Simple RAG Hallucination Rate", f"{rag_hall_rate:.0%}")
            st.info("Multi-Agent hallucination report unavailable for comparison.")
        else:
            st.info("Comparison unavailable — could not complete analysis.")

        st.session_state.chat_history.append({
            "query": query,
            "simple_response": rag.get("response", "No response.") if rag and "error" not in rag else "Error",
            "agent_response": agent.get("corrected_response", "No response.") if agent and "error" not in agent else "Error",
            "rag_hall_rate": rag_hall_rate if rag_available else None,
            "agent_hall_rate": agent_hall_rate if agent_available else None,
        })

if __name__ == "__main__":
    main()
