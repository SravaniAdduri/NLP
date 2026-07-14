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

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 12px;
    }
    h1 { font-size: 16px !important; font-weight: 700; }
    h2 { font-size: 16px !important; font-weight: 700; }
    h3 { font-size: 16px !important; font-weight: 700; }
    .stMarkdown p { font-size: 12px; }
    .stMarkdown li { font-size: 12px; }
    .stAlert p { font-size: 12px; }
    .stMetric label { font-size: 12px; }
    .stMetric [data-testid="metric-container"] div { font-size: 12px; }
    .stCaption { font-size: 11px; }
    code, pre { font-size: 12px; }
    .response-box, .response-box p, .response-box li, .response-box div {
        font-size: 12px !important;
        line-height: 1.5 !important;
    }
    .response-box h1, .response-box h2, .response-box h3,
    .response-box h4, .response-box h5, .response-box h6 {
        font-size: 16px !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.35rem !important;
    }
</style>
""", unsafe_allow_html=True)

EVALUATION_QUESTIONS = [
    "What encryption is used for data at rest in SecureMail?",
    "What is the session timeout in SecureMail v2.1?",
    "Does SecureMail support attachments in v2.1?",
    "What are the RPO and RTO for Billing API recovery?",
    "How often are incremental backups taken?",
    "What is the required customer communication frequency during incidents?",
    "What is the attendance requirement for each course?",
    "How many credits does the M.Tech thesis carry?",
    "What is the plagiarism review threshold?",
    "Compare SecureMail session timeout with Billing API alert acknowledgment target.",
    "Which document mentions a 30-minute expiry and what does it refer to?",
    "Is there a native SecureMail mobile app in v2.1?",
    "Does the runbook define SEV-4 incidents?",
    "Is the passing mark 35 out of 100 in the university handbook?",
]


def _answer_to_row(question, rag, agent, rag_verification, agent_report, agent_verification):
    rag_answer = rag.get("response", "") if rag and "error" not in rag else ""
    agent_answer = agent.get("corrected_response", "") if agent and "error" not in agent else ""
    rag_rate = rag_verification.get("hallucination_rate", 0.0) if rag_verification and not rag_verification.get("error") else None
    agent_rate = None
    if agent_report and agent_report.get("hallucination_rate") is not None:
        agent_rate = agent_report.get("hallucination_rate", 0.0)
    elif agent_verification and not agent_verification.get("error"):
        agent_rate = agent_verification.get("hallucination_rate", 0.0)

    verdict = "N/A"
    if rag_rate is not None and agent_rate is not None:
        verdict = "Multi-Agent" if agent_rate < rag_rate else ("Simple RAG" if rag_rate < agent_rate else "Tie")

    return {
        "question": question,
        "simple_answer": rag_answer,
        "multi_agent_answer": agent_answer,
        "simple_hallucination_rate": rag_rate,
        "multi_agent_hallucination_rate": agent_rate,
        "better_system": verdict,
        "simple_latency_ms": rag.get("latency_ms", None) if rag and "error" not in rag else None,
        "multi_agent_latency_ms": agent.get("metrics", {}).get("latency_ms", None) if agent and "error" not in agent else None,
    }


def _build_csv(rows):
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()) if rows else [])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")

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
            f'<div class="response-box" style="background:{color}; padding:10px; margin:6px 0; '
            f'border-radius:6px; border-left:5px solid {border};">'
            f'<b>{icon} {label}</b> (confidence: {conf:.2f})<br>{sent}</div>',
            unsafe_allow_html=True,
        )


def render_response_box(title: str, text: str, color: str = "#EAF3FF", border: str = "#4A78C2"):
    """Render response text with consistent body font and modest heading style."""
    safe_text = text.replace("\n", "<br>") if text else "No response."
    st.markdown(
        f'''
        <div class="response-box" style="background:{color}; padding:16px 18px; margin:8px 0; '
            f'border-radius:10px; border-left:5px solid {border}; color:#1f2937;">
            <div style="font-size:16px; font-weight:700; margin-bottom:8px;">{title}</div>
            <div style="font-size:12px; line-height:1.55;">{safe_text}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_compact_response(title: str, text: str, color: str, border: str):
    """Render a compact answer block with consistent typography."""
    safe_text = text if text else "No response."
    st.markdown(
        f'''
        <div class="response-box" style="background:{color}; padding:14px 16px; margin:8px 0; '
            f'border-radius:10px; border-left:5px solid {border}; color:#1f2937;">
            <div style="font-size:16px; font-weight:700; margin-bottom:8px;">{title}</div>
            <div style="font-size:12px; line-height:1.5; white-space:pre-wrap;">{safe_text}</div>
        </div>
        ''',
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
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = []

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

    upload_status = st.empty()
    action_col1, action_col2, action_col3 = st.columns([1, 1, 6])

    with action_col1:
        if st.button("🧹 Clear KB", key="btn_clear_kb", help="Clear all indexed documents"):
            result = api_reset_kb()
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.last_uploaded = set()
                st.session_state.chat_history = []
                st.session_state.evaluation_results = []
                upload_status.success("Knowledge base cleared.")
                st.rerun()

    if uploaded_files:
        if "last_uploaded" not in st.session_state:
            st.session_state.last_uploaded = set()
        current_names = {f.name for f in uploaded_files}
        new_files = [f for f in uploaded_files if f.name not in st.session_state.last_uploaded]
        if new_files:
            total_chunks = 0
            failures = []
            for f in new_files:
                with st.spinner(f"Indexing {f.name}..."):
                    result = api_upload_file(f)
                if "error" in result:
                    failures.append(f"{f.name}: {result['error']}")
                else:
                    total_chunks += int(result.get("num_chunks", 0))
            st.session_state.last_uploaded = current_names
            if failures:
                upload_status.error("Some files failed: " + ", ".join(failures))
            if total_chunks > 0:
                upload_status.success(f"✅ Indexed **{total_chunks}** chunks from {len(new_files)} file(s).")
                st.rerun()

        with action_col2:
            st.caption(f"Selected {len(uploaded_files)} file(s)")

    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

    with tab_chat:
        # === SECTION 2: CONVERSATIONAL CHAT ===
        st.subheader("💬 Chat")
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["query"])
            with st.chat_message("assistant"):
                render_response_box("Simple RAG Answer", turn.get("simple_response", "No response."), "#EAF3FF", "#4A78C2")
                render_response_box("Multi-Agent Answer", turn.get("agent_response", "No response."), "#E9FFF1", "#2E8B57")

                if turn.get("rag_hall_rate") is not None and turn.get("agent_hall_rate") is not None:
                    col_a, col_b = st.columns(2)
                    col_a.metric("Simple Hallucination", f"{turn['rag_hall_rate']:.0%}")
                    col_b.metric("Multi-Agent Hallucination", f"{turn['agent_hall_rate']:.0%}")

        query = st.chat_input("Ask anything about your uploaded documents...")

        if query:
            with st.chat_message("user"):
                st.write(query)

            st.markdown("---")
            col_rag, col_agent = st.columns(2)

            rag = None
            agent = None

            with col_rag:
                st.subheader("📄 Simple RAG")
                with st.spinner("Retrieving..."):
                    rag = api_simple_rag(query)
                if "error" in rag:
                    st.error(rag["error"])
                else:
                    render_compact_response("Answer", rag.get("response", "No response."), "#EAF3FF", "#4A78C2")
                    st.caption(f"⏱️ {rag.get('latency_ms', 0):.0f} ms | Method: Direct vector search (no reranking)")
                    render_evidence(rag.get("evidence", []), rag.get("evidence_scores", []))

            with col_agent:
                st.subheader("🤖 Multi-Agent Pipeline")
                with st.spinner("Running multi-agent pipeline..."):
                    agent = api_multi_agent(query)
                if "error" in agent:
                    st.error(agent["error"])
                else:
                    agent_response = agent.get("corrected_response") or "No response."
                    render_compact_response("Corrected Answer", agent_response, "#E9FFF1", "#2E8B57")
                    latency = agent.get("metrics", {}).get("latency_ms", 0) if agent.get("metrics") else 0
                    st.caption(f"⏱️ {latency:.0f} ms | Method: Query rewrite → Multi-query retrieval → Rerank → Generate")
                    render_evidence(agent.get("ranked_evidence", []))
                    if agent.get("citations"):
                        with st.expander("📑 Citations"):
                            for c in agent["citations"]:
                                st.markdown(f"**[{c['id']}]** {c['text']}")

            rag_verification = None
            agent_verification = None
            agent_native_report = None
            with st.spinner("Running hallucination analysis..."):
                if rag and "error" not in rag and rag.get("response"):
                    rag_verification = api_verify_response(rag["response"], rag.get("evidence", []))

                if agent and "error" not in agent and agent.get("corrected_response"):
                    agent_native_report = agent.get("hallucination_report")
                    if not (agent_native_report and agent_native_report.get("sentence_results")):
                        agent_verification = api_verify_response(
                            agent.get("corrected_response", ""),
                            agent.get("generated_evidence") or agent.get("ranked_evidence", []),
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

    with tab_eval:
        st.subheader("📊 Batch Evaluation")
        st.caption("Run a preset list of questions to compare Simple RAG and Multi-Agent side by side and export results.")

        selected_questions = st.multiselect(
            "Select questions to run",
            options=EVALUATION_QUESTIONS,
            default=EVALUATION_QUESTIONS[:6],
        )

        eval_col1, eval_col2 = st.columns([1, 1])
        with eval_col1:
            run_eval = st.button("▶ Run Evaluation", type="primary", use_container_width=True)
        with eval_col2:
            clear_eval = st.button("🧹 Clear Results", use_container_width=True)

        if clear_eval:
            st.session_state.evaluation_results = []
            st.success("Evaluation results cleared.")

        if run_eval and selected_questions:
            results = []
            progress = st.progress(0)
            status = st.empty()
            for idx, question in enumerate(selected_questions, 1):
                status.write(f"Running {idx}/{len(selected_questions)}: {question}")
                rag = api_simple_rag(question)
                agent = api_multi_agent(question)

                rag_verification = None
                if rag and "error" not in rag and rag.get("response"):
                    rag_verification = api_verify_response(rag["response"], rag.get("evidence", []))

                agent_native_report = agent.get("hallucination_report") if agent and "error" not in agent else None
                agent_verification = None
                if not (agent_native_report and agent_native_report.get("sentence_results")) and agent and "error" not in agent and agent.get("corrected_response"):
                    agent_verification = api_verify_response(
                        agent.get("corrected_response", ""),
                        agent.get("generated_evidence") or agent.get("ranked_evidence", []),
                    )

                row = _answer_to_row(question, rag, agent, rag_verification, agent_native_report, agent_verification)
                results.append(row)
                progress.progress(idx / len(selected_questions))

            st.session_state.evaluation_results = results
            status.success("Evaluation complete.")

        if st.session_state.evaluation_results:
            eval_rows = st.session_state.evaluation_results
            simple_rates = [r["simple_hallucination_rate"] for r in eval_rows if r["simple_hallucination_rate"] is not None]
            multi_rates = [r["multi_agent_hallucination_rate"] for r in eval_rows if r["multi_agent_hallucination_rate"] is not None]

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Questions Run", len(eval_rows))
            metric_col2.metric("Avg Simple Hallucination", f"{(sum(simple_rates)/len(simple_rates)):.0%}" if simple_rates else "N/A")
            metric_col3.metric("Avg Multi-Agent Hallucination", f"{(sum(multi_rates)/len(multi_rates)):.0%}" if multi_rates else "N/A")

            st.dataframe(eval_rows, use_container_width=True)

            csv_bytes = _build_csv(eval_rows)
            st.download_button(
                "⬇ Download CSV",
                data=csv_bytes,
                file_name="rag_evaluation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        return

if __name__ == "__main__":
    main()
