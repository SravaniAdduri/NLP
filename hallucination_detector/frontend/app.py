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

# --- MAIN PAGE ---

def main():
    st.title("🔍 Hallucination Detection & Fact Verification")
    st.caption("Multi-Agent RAG Framework — Compare Simple RAG vs Multi-Agent Pipeline")

    health = api_health()
    if health["status"] == "offline":
        st.error("❌ Backend API offline. Start with: `uvicorn api.main:app --reload --port 8000`")
        return

    # === SECTION 1: UPLOAD DOCUMENT ===
    st.subheader("📄 Upload Document")
    col_upload, col_status = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a PDF, TXT, MD, or HTML file",
            type=["pdf", "txt", "md", "html"],
            label_visibility="collapsed",
        )
    with col_status:
        chunks = health.get("index_size", 0)
        if chunks > 0:
            st.success(f"✅ **{chunks}** chunks indexed")
        else:
            st.warning("No document yet")

    if uploaded_file:
        st.caption(f"📎 Selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        if st.button("📤 Upload & Index", key="btn_upload", type="primary"):
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                result = api_upload_file(uploaded_file)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"✅ Done! **{result['num_chunks']}** chunks created from {uploaded_file.name}")
                st.rerun()

    has_docs = health.get("index_size", 0) > 0
    if not has_docs:
        st.info("⬆️ Upload a document first, then ask questions below.")
        return

    st.markdown("---")

    # === SECTION 2: ASK A QUESTION ===
    st.subheader("💬 Ask a Question")
    query = st.text_input(
        "Type your question:",
        placeholder="e.g., What is the main purpose of this document?",
        key="main_query",
    )

    if st.button("📩 Send", type="primary", disabled=not query, key="btn_send"):
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
                st.markdown("**Answer:**")
                st.info(rag.get("response", "No response."))
                st.caption(f"⏱️ {rag.get('latency_ms', 0):.0f} ms")
                render_evidence(rag.get("evidence", []), rag.get("evidence_scores", []))

        with col_agent:
            st.subheader("🤖 Multi-Agent Pipeline")
            with st.spinner("Running pipeline..."):
                agent = api_multi_agent(query)
            if "error" in agent:
                st.error(agent["error"])
            else:
                st.markdown("**Answer:**")
                agent_response = agent.get("corrected_response") or "No response."
                st.success(agent_response)
                latency = agent.get("metrics", {}).get("latency_ms", 0) if agent.get("metrics") else 0
                st.caption(f"⏱️ {latency:.0f} ms")
                render_evidence(agent.get("ranked_evidence", []))
                if agent.get("citations"):
                    with st.expander("📑 Citations"):
                        for c in agent["citations"]:
                            st.markdown(f"**[{c['id']}]** {c['text']}")

        # Automatic hallucination analysis for both generated responses
        rag_verification = None
        agent_verification = None
        with st.spinner("Running hallucination analysis for both responses..."):
            if rag and "error" not in rag and rag.get("response"):
                rag_verification = api_verify_response(
                    rag["response"],
                    rag.get("evidence", []),
                )
            if agent and "error" not in agent and agent.get("corrected_response"):
                agent_verification = api_verify_response(
                    agent["corrected_response"],
                    agent.get("ranked_evidence", []),
                )

        st.markdown("---")
        st.subheader("🔬 Hallucination Analysis (Auto)")

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
            st.subheader("🤖 Multi-Agent")
            if agent_verification and "error" not in agent_verification and agent_verification.get("sentence_results"):
                agent_available = True
                av = agent_verification
                agent_hall_rate = av.get("hallucination_rate", 0.0)
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Supported", av.get("supported_count", 0))
                c2.metric("❌ Contradicted", av.get("contradicted_count", 0))
                c3.metric("⚠️ Uncertain", av.get("neutral_count", 0))
                st.metric("Hallucination Rate", f"{agent_hall_rate:.0%}")
                render_sentence_highlight(av.get("sentence_results", []))
            else:
                st.info("Could not analyze Multi-Agent response.")

        st.markdown("---")
        st.subheader("🏁 Which Hallucinates More?")
        if rag_available and agent_available:
            if rag_hall_rate > agent_hall_rate:
                diff = rag_hall_rate - agent_hall_rate
                st.error(
                    f"Simple RAG hallucinates more: **{rag_hall_rate:.0%}** vs "
                    f"Multi-Agent **{agent_hall_rate:.0%}** (difference **{diff:.0%}**)."
                )
            elif agent_hall_rate > rag_hall_rate:
                diff = agent_hall_rate - rag_hall_rate
                st.warning(
                    f"Multi-Agent hallucinates more: **{agent_hall_rate:.0%}** vs "
                    f"Simple RAG **{rag_hall_rate:.0%}** (difference **{diff:.0%}**)."
                )
            else:
                st.success(f"Both are equal at **{rag_hall_rate:.0%}** hallucination rate.")

            c1, c2 = st.columns(2)
            c1.metric("Simple RAG Hallucination", f"{rag_hall_rate:.0%}")
            c2.metric("Multi-Agent Hallucination", f"{agent_hall_rate:.0%}")
        else:
            st.info("Comparison verdict unavailable because one or both analyses could not be completed.")

    st.markdown("---")

    # === SECTION 3: HALLUCINATION ANALYSIS ===
    st.subheader("🔬 Hallucination Analysis")
    st.caption("Paste an AI-generated response to check it for hallucinations against the uploaded document")

    llm_text = st.text_area(
        "LLM Response to verify:",
        placeholder="Paste any AI-generated text here...",
        height=120,
        key="llm_input",
    )

    if st.button("📩 Analyze", type="primary", disabled=not (query and llm_text), key="btn_analyze"):
        st.markdown("---")

        # Run both analyses
        with st.spinner("Running hallucination analysis on both pipelines..."):
            # Multi-Agent analysis (includes hallucination detection)
            result_agent = api_multi_agent(query, llm_text)
            # Simple RAG response
            rag = api_simple_rag(query)
            # Also verify the Simple RAG response for hallucinations
            rag_verification = None
            if "error" not in rag and rag.get("response"):
                try:
                    r = requests.post(f"{API_BASE_URL}/verify_response",
                        json={"text": rag["response"], "evidence": rag.get("evidence", [])}, timeout=120)
                    if r.status_code == 200:
                        rag_verification = r.json()
                except Exception:
                    pass

        # Show corrected responses side by side
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("📄 Simple RAG Response")
            if "error" not in rag:
                st.info(rag.get("response", "No response."))
            else:
                st.error(rag["error"])
        with col_right:
            st.subheader("🤖 Multi-Agent Corrected Response")
            if "error" not in result_agent:
                st.success(result_agent.get("corrected_response", "No response."))
            else:
                st.error(result_agent["error"])
                return

        st.markdown("---")

        # Hallucination analysis for BOTH
        col_h1, col_h2 = st.columns(2)

        # Simple RAG hallucination report
        rag_hall_rate = 0.0
        with col_h1:
            st.subheader("📄 Simple RAG — Hallucination Report")
            if rag_verification and rag_verification.get("sentence_results"):
                rv = rag_verification
                rag_hall_rate = rv["hallucination_rate"]
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Supported", rv["supported_count"])
                c2.metric("❌ Contradicted", rv["contradicted_count"])
                c3.metric("⚠️ Uncertain", rv["neutral_count"])
                
                # Show calculation
                total = rv["total_sentences"]
                contradicted = rv["contradicted_count"]
                uncertain = rv["neutral_count"]
                hallucinated = contradicted + uncertain
                
                st.markdown(f"""
**Hallucination Calculation:**
```
Hallucination Rate = (Contradicted + Uncertain) / Total Sentences
                   = ({contradicted} + {uncertain}) / {total}
                   = {hallucinated} / {total}
                   = {rag_hall_rate:.0%}
```
                """)
                st.metric("Hallucination Rate", f"{rag_hall_rate:.0%}")
                st.markdown("---")
                render_sentence_highlight(rv["sentence_results"])
            else:
                st.info("Could not analyze Simple RAG response.")

        # Multi-Agent hallucination report
        agent_hall_rate = 0.0
        with col_h2:
            st.subheader("🤖 Multi-Agent — Hallucination Report")
            report = result_agent.get("hallucination_report") if "error" not in result_agent else None
            if report and report.get("sentence_results"):
                agent_hall_rate = report["hallucination_rate"]
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Supported", report["supported_count"])
                c2.metric("❌ Contradicted", report["contradicted_count"])
                c3.metric("⚠️ Uncertain", report["neutral_count"])
                
                # Show calculation
                total = report["total_sentences"]
                contradicted = report["contradicted_count"]
                uncertain = report["neutral_count"]
                hallucinated = contradicted + uncertain
                
                st.markdown(f"""
**Hallucination Calculation:**
```
Hallucination Rate = (Contradicted + Uncertain) / Total Sentences
                   = ({contradicted} + {uncertain}) / {total}
                   = {hallucinated} / {total}
                   = {agent_hall_rate:.0%}
```
                """)
                st.metric("Hallucination Rate", f"{agent_hall_rate:.0%}")
                st.markdown("---")
                render_sentence_highlight(report["sentence_results"])
            else:
                st.info("Could not analyze Multi-Agent response.")

        # === VERDICT: Which is more hallucinating? ===
        st.markdown("---")
        st.subheader("🏆 Verdict")
        if rag_hall_rate > 0 or agent_hall_rate > 0:
            if rag_hall_rate > agent_hall_rate:
                diff = rag_hall_rate - agent_hall_rate
                st.error(
                    f"**Simple RAG is more hallucinating** (rate: {rag_hall_rate:.0%}) "
                    f"compared to Multi-Agent Pipeline (rate: {agent_hall_rate:.0%}). "
                    f"The Multi-Agent system reduces hallucinations by **{diff:.0%}**."
                )
            elif agent_hall_rate > rag_hall_rate:
                diff = agent_hall_rate - rag_hall_rate
                st.warning(
                    f"**Multi-Agent Pipeline is more hallucinating** (rate: {agent_hall_rate:.0%}) "
                    f"compared to Simple RAG (rate: {rag_hall_rate:.0%})."
                )
            else:
                st.info(
                    f"Both systems have the **same hallucination rate** ({rag_hall_rate:.0%}). "
                    f"No significant difference detected."
                )

            # Comparison table
            st.markdown("---")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Simple RAG Hallucination Rate", f"{rag_hall_rate:.0%}")
            col_m2.metric("Multi-Agent Hallucination Rate", f"{agent_hall_rate:.0%}")
        else:
            st.success("✅ Neither system shows significant hallucination for this input.")

        # Show methodology explanation
        show_hallucination_methodology()

        # Additional tabs for detailed info
        if "error" not in result_agent:
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["✅ Fact Verification", "📚 Evidence", "📊 Metrics"])

            with tab1:
                claims = result_agent.get("verified_claims", [])
                if claims:
                    for cl in claims:
                        icon = {"VERIFIED": "✅", "REFUTED": "❌", "UNVERIFIABLE": "⚠️"}.get(cl["verdict"], "❓")
                        st.markdown(f"{icon} **{cl['verdict']}** ({cl['confidence']:.2f}): {cl['claim']}")
                vr = result_agent.get("verification_report")
                if vr:
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Verified", vr["verified_count"])
                    c2.metric("Refuted", vr["refuted_count"])
                    c3.metric("Accuracy", f"{vr['overall_accuracy']:.0%}")

            with tab2:
                render_evidence(result_agent.get("ranked_evidence", []))

            with tab3:
                m = result_agent.get("metrics")
                if m:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Precision", f"{m.get('precision', 0):.3f}")
                    c2.metric("Recall", f"{m.get('recall', 0):.3f}")
                    c3.metric("F1", f"{m.get('f1_score', 0):.3f}")
                    c4.metric("Latency", f"{m.get('latency_ms', 0):.0f} ms")


if __name__ == "__main__":
    main()
