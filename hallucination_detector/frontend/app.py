"""
Streamlit Frontend Application
Provides an interactive UI for:
- Uploading documents (PDF, text)
- Asking questions
- Viewing retrieved evidence
- Highlighting hallucinated sentences
- Displaying the corrected answer with citations
- Viewing evaluation metrics
"""

import streamlit as st
import requests
import json
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="Hallucination Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def check_api_health() -> dict:
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "index_size": 0, "message": "Backend API not running"}
    except Exception as e:
        return {"status": "error", "index_size": 0, "message": str(e)}


def upload_file(file) -> dict:
    """Upload a file to the backend."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload/file", files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"Upload failed: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


def upload_text(text: str, source_name: str) -> dict:
    """Upload raw text to the backend."""
    try:
        payload = {"text": text, "source_name": source_name}
        response = requests.post(f"{API_BASE_URL}/upload/text", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def upload_url(url: str) -> dict:
    """Ingest a URL into the knowledge base."""
    try:
        payload = {"url": url}
        response = requests.post(f"{API_BASE_URL}/upload/url", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def process_query(query: str, llm_response: str = "") -> dict:
    """Send a query to the multi-agent pipeline backend."""
    try:
        payload = {"query": query, "llm_response": llm_response}
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"Query failed: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


def simple_rag_query(query: str) -> dict:
    """Send a query to the simple RAG endpoint."""
    try:
        payload = {"query": query, "llm_response": ""}
        response = requests.post(f"{API_BASE_URL}/simple_rag", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"Simple RAG failed: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


def get_system_status() -> dict:
    """Get detailed system status."""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        return response.json()
    except Exception:
        return {"llm_active": False, "llm_provider": "unknown", "index_size": 0}


def render_sidebar():
    """Render the sidebar with system status and document upload."""
    with st.sidebar:
        st.title("📚 Knowledge Base")

        # Health check
        health = check_api_health()
        if health["status"] == "healthy":
            st.success(f"✅ API Online | {health['index_size']} chunks indexed")

            # Show LLM status
            status = get_system_status()
            if status.get("llm_active"):
                st.success(f"🤖 LLM: **{status.get('llm_provider', 'active')}**")
            else:
                st.warning("⚠️ LLM: OFF (extractive mode)\nSet HF_TOKEN in .env for better responses")
        elif health["status"] == "offline":
            st.error("❌ Backend API offline. Start with: `uvicorn api.main:app --reload`")
            return
        else:
            st.warning(f"⚠️ {health['message']}")

        st.markdown("---")

        # File upload
        st.subheader("📄 Upload Document")
        st.caption("Supported: PDF, TXT, MD, HTML (any size)")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "md", "html"],
            help="Upload PDF, text, markdown, or HTML files. File is processed into chunks for search.",
        )
        if uploaded_file:
            st.info(f"Selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
            if st.button("📤 Upload & Index File", key="upload_file", type="primary"):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    result = upload_file(uploaded_file)
                    if "error" in result:
                        st.error(f"Upload failed: {result['error']}")
                    else:
                        st.success(f"✅ Indexed! {result['num_chunks']} chunks created from {uploaded_file.name}")
                        st.balloons()

        st.markdown("---")

        # Text input
        st.subheader("📝 Paste Text")
        text_input = st.text_area("Paste text content:", height=100, key="text_input")
        if text_input and st.button("📝 Index Text", key="add_text"):
            with st.spinner("Indexing text..."):
                result = upload_text(text_input, "pasted_text")
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✅ Indexed! {result['num_chunks']} chunks created")

        st.markdown("---")
        url_input = st.text_input("Enter URL:", key="url_input")
        if url_input and st.button("🌐 Ingest URL", key="ingest_url"):
            with st.spinner("Fetching and processing URL..."):
                result = upload_url(url_input)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✅ {result['message']} ({result['num_chunks']} chunks)")


def render_hallucination_highlight(sentence_results: list):
    """Render sentences with color-coded hallucination highlighting."""
    for result in sentence_results:
        sentence = result["sentence"]
        label = result["label"]
        confidence = result["confidence"]

        if label == "SUPPORTED":
            color = "#d4edda"  # Green
            icon = "✅"
        elif label == "CONTRADICTED":
            color = "#f8d7da"  # Red
            icon = "❌"
        else:
            color = "#fff3cd"  # Yellow
            icon = "⚠️"

        st.markdown(
            f'<div style="background-color: {color}; padding: 8px; margin: 4px 0; '
            f'border-radius: 4px; border-left: 4px solid {"#28a745" if label == "SUPPORTED" else "#dc3545" if label == "CONTRADICTED" else "#ffc107"};">'
            f'{icon} <strong>{label}</strong> (confidence: {confidence:.2f})<br>'
            f'{sentence}</div>',
            unsafe_allow_html=True,
        )

        if result.get("supporting_evidence") and label != "SUPPORTED":
            with st.expander(f"Evidence for: \"{sentence[:50]}...\""):
                st.text(result["supporting_evidence"])


def render_metrics(metrics: dict):
    """Render evaluation metrics in a dashboard layout."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    with col2:
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    with col3:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
    with col4:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("Hallucination Rate", f"{metrics.get('hallucination_rate', 0):.3f}")
    with col6:
        st.metric("Latency (ms)", f"{metrics.get('latency_ms', 0):.0f}")
    with col7:
        st.metric("Claims Verified", metrics.get("num_claims_verified", 0))
    with col8:
        st.metric("Avg Confidence", f"{metrics.get('average_confidence', 0):.3f}")


def render_main():
    """Render the main content area."""
    st.title("🔍 Hallucination Detection & Fact Verification")
    st.caption("Multi-Agent RAG Framework — Compare Simple RAG vs Multi-Agent responses")

    # Check if documents are loaded
    health = check_api_health()
    has_docs = health.get("index_size", 0) > 0

    if not has_docs:
        st.warning("⚠️ **No documents indexed yet.** Upload a document using the sidebar first, then ask questions.")

    # Query input
    st.subheader("Ask a Question")
    query = st.text_input(
        "Your question:",
        placeholder="e.g., What is the main purpose of this document?",
        key="main_query",
    )

    # Get Answer button
    if st.button("🚀 Get Answer (Side-by-Side Comparison)", type="primary", disabled=(not query or not has_docs)):
        st.markdown("---")

        # Run both models in parallel display
        col_rag, col_agent = st.columns(2)

        # --- LEFT: Simple RAG ---
        with col_rag:
            st.subheader("📄 Simple RAG")
            with st.spinner("Running Simple RAG..."):
                rag_result = simple_rag_query(query)

            if "error" in rag_result:
                st.error(rag_result["error"])
            else:
                st.markdown(f"**Response:**")
                st.info(rag_result.get("response", "No response."))
                st.caption(f"⏱️ Latency: {rag_result.get('latency_ms', 0):.0f} ms")

                with st.expander("📚 Evidence Used"):
                    for i, ev in enumerate(rag_result.get("evidence", [])[:3], 1):
                        score = rag_result.get("evidence_scores", [])[i-1] if i-1 < len(rag_result.get("evidence_scores", [])) else 0
                        st.markdown(f"**[{i}]** (score: {score:.2f})")
                        st.text(ev[:250])
                        st.markdown("---")

        # --- RIGHT: Multi-Agent ---
        with col_agent:
            st.subheader("🤖 Multi-Agent Pipeline")
            with st.spinner("Running Multi-Agent Pipeline..."):
                agent_result = process_query(query, "")

            if "error" in agent_result:
                st.error(agent_result["error"])
            else:
                st.markdown(f"**Response:**")
                st.success(agent_result.get("corrected_response", "No response."))
                latency = agent_result.get("metrics", {}).get("latency_ms", 0) if agent_result.get("metrics") else 0
                st.caption(f"⏱️ Latency: {latency:.0f} ms")

                with st.expander("📚 Evidence Used"):
                    for i, ev in enumerate(agent_result.get("ranked_evidence", [])[:3], 1):
                        st.markdown(f"**[{i}]**")
                        st.text(ev[:250])
                        st.markdown("---")

                if agent_result.get("citations"):
                    with st.expander("📖 Citations"):
                        for c in agent_result["citations"]:
                            st.markdown(f"**[{c['id']}]** {c['text']}")

    # --- Full Hallucination Analysis Section ---
    st.markdown("---")
    st.subheader("🔬 Full Hallucination Analysis")
    st.caption("Paste an LLM-generated response to verify it against the knowledge base")

    llm_response = st.text_area(
        "LLM Response to verify:",
        placeholder="Paste any AI-generated response here to check for hallucinations...",
        height=120,
        key="llm_response",
    )

    if st.button("🔍 Run Hallucination Analysis", disabled=not (query and llm_response)):
        with st.spinner("Running full multi-agent analysis..."):
            result = process_query(query, llm_response)

        if "error" in result:
            st.error(f"Error: {result['error']}")
            return

        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Corrected Response",
            "🔴 Hallucination Analysis",
            "📚 Evidence",
            "✅ Fact Verification",
            "📊 Metrics",
        ])

        with tab1:
            col_orig, col_corrected = st.columns(2)
            with col_orig:
                st.markdown("**❌ Original (with potential hallucinations):**")
                st.warning(llm_response)
            with col_corrected:
                st.markdown("**✅ Corrected (grounded in evidence):**")
                st.success(result.get("corrected_response", "No response generated."))

            if result.get("citations"):
                st.subheader("Citations")
                for citation in result["citations"]:
                    st.markdown(f"**[{citation['id']}]** {citation['text']}")

        with tab2:
            st.subheader("Hallucination Analysis")
            report = result.get("hallucination_report")
            if report and report.get("sentence_results"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("✅ Supported", report["supported_count"])
                with col2:
                    st.metric("❌ Contradicted", report["contradicted_count"])
                with col3:
                    st.metric("⚠️ Uncertain", report["neutral_count"])
                with col4:
                    st.metric("Hallucination Rate", f"{report['hallucination_rate']:.0%}")

                st.markdown("---")
                st.subheader("Sentence-Level Analysis")
                render_hallucination_highlight(report.get("sentence_results", []))
            else:
                st.warning("Could not perform hallucination analysis.")

        with tab3:
            st.subheader("Retrieved Evidence")
            evidence_list = result.get("ranked_evidence", [])
            if evidence_list:
                for i, ev in enumerate(evidence_list, 1):
                    with st.expander(f"Evidence [{i}]: {ev[:80]}..."):
                        st.write(ev)
            else:
                st.warning("No evidence retrieved.")

        with tab4:
            st.subheader("Fact Verification")
            claims = result.get("verified_claims", [])
            if claims:
                for claim in claims:
                    verdict_icon = {"VERIFIED": "✅", "REFUTED": "❌", "UNVERIFIABLE": "⚠️"}.get(claim["verdict"], "❓")
                    st.markdown(f"{verdict_icon} **{claim['verdict']}** (confidence: {claim['confidence']:.2f}): {claim['claim']}")

                    if claim.get("supporting_evidence"):
                        with st.expander("Supporting evidence"):
                            for ev in claim["supporting_evidence"]:
                                st.text(ev[:200])
            else:
                st.info("No verifiable claims found.")

            vr = result.get("verification_report")
            if vr:
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Verified", vr["verified_count"])
                with col2:
                    st.metric("Refuted", vr["refuted_count"])
                with col3:
                    st.metric("Accuracy", f"{vr['overall_accuracy']:.0%}")

        with tab5:
            st.subheader("Evaluation Metrics")
            metrics = result.get("metrics")
            if metrics:
                render_metrics(metrics)
            else:
                st.info("Metrics unavailable.")

        if result.get("errors"):
            with st.expander("⚠️ Warnings"):
                for error in result["errors"]:
                    st.warning(error)


def main():
    """Main entry point for the Streamlit app."""
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
