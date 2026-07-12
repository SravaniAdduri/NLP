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
    """Send a query to the backend for processing."""
    try:
        payload = {"query": query, "llm_response": llm_response}
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"Query failed: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


def render_sidebar():
    """Render the sidebar with system status and document upload."""
    with st.sidebar:
        st.title("📚 Knowledge Base")

        # Health check
        health = check_api_health()
        if health["status"] == "healthy":
            st.success(f"✅ API Online | {health['index_size']} chunks indexed")
        elif health["status"] == "offline":
            st.error("❌ Backend API offline. Start with: `uvicorn api.main:app --reload`")
            return
        else:
            st.warning(f"⚠️ {health['message']}")

        st.markdown("---")

        # File upload
        st.subheader("Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "md", "html"],
            help="Upload PDF, text, markdown, or HTML files",
        )
        if uploaded_file and st.button("📤 Upload File", key="upload_file"):
            with st.spinner("Processing document..."):
                result = upload_file(uploaded_file)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✅ {result['message']} ({result['num_chunks']} chunks)")

        st.markdown("---")

        # Text input
        st.subheader("Add Text")
        text_input = st.text_area("Paste text content:", height=100, key="text_input")
        source_name = st.text_input("Source name:", value="manual_input", key="source_name")
        if text_input and st.button("📝 Add Text", key="add_text"):
            with st.spinner("Ingesting text..."):
                result = upload_text(text_input, source_name)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✅ Added {result['num_chunks']} chunks")

        st.markdown("---")

        # URL input
        st.subheader("Add Web Page")
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
    st.caption("Multi-Agent RAG Framework for detecting hallucinations in LLM responses")

    # Query input
    st.subheader("Ask a Question")
    query = st.text_input(
        "Your question:",
        placeholder="e.g., What are the main causes of climate change?",
        key="main_query",
    )

    # Two columns for the two actions
    col_ask, col_analyze = st.columns(2)

    with col_ask:
        ask_clicked = st.button("💬 Get Answer", type="secondary", disabled=not query)

    with col_analyze:
        analyze_clicked = st.button("🚀 Analyze for Hallucinations", type="primary", disabled=not query)

    # LLM response input (only shown when analyze is the intent)
    llm_response = ""
    if analyze_clicked or st.session_state.get("show_llm_input", False):
        st.session_state["show_llm_input"] = True
        st.markdown("---")
        st.subheader("Paste an LLM Response to Verify")
        llm_response = st.text_area(
            "LLM Response:",
            placeholder="Paste the LLM-generated response you want to fact-check...",
            height=150,
            key="llm_response",
        )
        run_analysis = st.button("🔍 Run Full Analysis", type="primary", disabled=not (query and llm_response))
    else:
        run_analysis = False

    # --- MODE 1: Simple Q&A (Get Answer) ---
    if ask_clicked and query:
        st.session_state["show_llm_input"] = False
        with st.spinner("Retrieving answer from knowledge base..."):
            result = process_query(query, "")

        if "error" in result:
            st.error(f"Error: {result['error']}")
            return

        st.markdown("---")
        st.subheader("📋 Answer")
        st.write(result.get("corrected_response", "No response generated."))

        if result.get("citations"):
            with st.expander("📚 Sources"):
                for citation in result["citations"]:
                    st.markdown(f"**[{citation['id']}]** {citation['text']}")

        if result.get("ranked_evidence"):
            with st.expander(f"📖 Retrieved Evidence ({len(result['ranked_evidence'])} passages)"):
                for i, ev in enumerate(result["ranked_evidence"], 1):
                    st.markdown(f"**[{i}]** {ev[:300]}{'...' if len(ev) > 300 else ''}")
                    st.markdown("---")

        if result.get("metrics") and result["metrics"].get("latency_ms"):
            st.caption(f"⏱️ Response time: {result['metrics']['latency_ms']:.0f} ms")

    # --- MODE 2: Full Hallucination Analysis ---
    if run_analysis and query and llm_response:
        with st.spinner("Running full multi-agent analysis pipeline..."):
            result = process_query(query, llm_response)

        if "error" in result:
            st.error(f"Error: {result['error']}")
            return

        st.markdown("---")

        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Corrected Response",
            "🔴 Hallucination Analysis",
            "📚 Evidence",
            "✅ Fact Verification",
            "📊 Metrics",
        ])

        with tab1:
            st.subheader("Original LLM Response")
            st.warning(llm_response)

            st.subheader("Corrected Response")
            st.success(result.get("corrected_response", "No response generated."))

            if result.get("citations"):
                st.subheader("Citations")
                for citation in result["citations"]:
                    st.markdown(f"**[{citation['id']}]** {citation['text']}")

        with tab2:
            st.subheader("Hallucination Analysis")
            report = result.get("hallucination_report")
            if report and report.get("sentence_results"):
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Supported", report["supported_count"])
                with col2:
                    st.metric("Contradicted", report["contradicted_count"])
                with col3:
                    st.metric("Not Enough Evidence", report["neutral_count"])

                st.metric("Hallucination Rate", f"{report['hallucination_rate']:.2%}")

                st.markdown("---")
                st.subheader("Sentence-Level Analysis")
                render_hallucination_highlight(report.get("sentence_results", []))
            else:
                st.warning("Could not perform hallucination analysis. Ensure documents are uploaded to the knowledge base.")

        with tab3:
            st.subheader("Retrieved Evidence")
            evidence_list = result.get("ranked_evidence", [])
            if evidence_list:
                for i, ev in enumerate(evidence_list, 1):
                    with st.expander(f"Evidence [{i}]: {ev[:80]}..."):
                        st.write(ev)
            else:
                st.warning("No evidence retrieved. Upload documents to the knowledge base first.")

        with tab4:
            st.subheader("Fact Verification")
            claims = result.get("verified_claims", [])
            if claims:
                for claim in claims:
                    verdict_icon = {
                        "VERIFIED": "✅",
                        "REFUTED": "❌",
                        "UNVERIFIABLE": "⚠️",
                    }.get(claim["verdict"], "❓")

                    st.markdown(
                        f"{verdict_icon} **{claim['verdict']}** "
                        f"(confidence: {claim['confidence']:.2f}): {claim['claim']}"
                    )

                    if claim.get("supporting_evidence"):
                        with st.expander("Supporting evidence"):
                            for ev in claim["supporting_evidence"]:
                                st.text(ev[:200])
            else:
                st.info("No verifiable claims found in the LLM response.")

            vr = result.get("verification_report")
            if vr:
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Verified", vr["verified_count"])
                with col2:
                    st.metric("Refuted", vr["refuted_count"])
                with col3:
                    st.metric("Overall Accuracy", f"{vr['overall_accuracy']:.2%}")

        with tab5:
            st.subheader("Evaluation Metrics")
            metrics = result.get("metrics")
            if metrics:
                render_metrics(metrics)
            else:
                st.info("Metrics unavailable.")

        # Show errors if any
        if result.get("errors"):
            with st.expander("⚠️ Warnings/Errors"):
                for error in result["errors"]:
                    st.warning(error)


def main():
    """Main entry point for the Streamlit app."""
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
