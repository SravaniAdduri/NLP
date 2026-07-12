# Adaptive Multi-Agent Retrieval-Augmented Framework for Hallucination Detection and Fact Verification

## System Architecture

This project implements a multi-agent system using LangGraph orchestration to detect hallucinations in LLM-generated responses and verify facts against a local knowledge base.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT FRONTEND                               │
│  [Upload PDFs] [Ask Questions] [View Evidence] [Highlight Hallucinations]│
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ HTTP/REST
┌──────────────────────────────────▼───────────────────────────────────────┐
│                          FASTAPI BACKEND                                  │
│  /api/query  /api/upload  /api/verify  /api/evaluate                     │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                     LANGGRAPH ORCHESTRATOR                                │
│                                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐             │
│  │   Query     │───▶│  Retrieval  │───▶│ Evidence Ranking │             │
│  │Understanding│    │    Agent    │    │      Agent       │             │
│  │   Agent     │    │             │    │                  │             │
│  └─────────────┘    └─────────────┘    └────────┬─────────┘             │
│                                                  │                        │
│  ┌─────────────┐    ┌─────────────┐    ┌────────▼─────────┐             │
│  │  Response   │◀───│    Fact     │◀───│  Hallucination   │             │
│  │ Generation  │    │Verification │    │   Detection      │             │
│  │   Agent     │    │    Agent    │    │     Agent        │             │
│  └──────┬──────┘    └─────────────┘    └──────────────────┘             │
│         │                                                                 │
│  ┌──────▼──────┐                                                         │
│  │ Evaluation  │                                                         │
│  │    Agent    │                                                         │
│  └─────────────┘                                                         │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                        DATA LAYER                                         │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │   FAISS    │  │   SQLite/    │  │  Document   │  │  HuggingFace   │  │
│  │Vector Store│  │  PostgreSQL  │  │   Store     │  │    Models      │  │
│  └────────────┘  └──────────────┘  └─────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
User Query
    │
    ▼
[Query Understanding Agent]
    │ Extracts entities, keywords, rewrites query
    ▼
[Retrieval Agent]
    │ Searches FAISS vector store
    │ Returns candidate documents
    ▼
[Evidence Ranking Agent]
    │ Cross-encoder reranking
    │ Deduplication, top-k selection
    ▼
[Hallucination Detection Agent]
    │ Sentence-level comparison with evidence
    │ Labels: Supported / Contradicted / Not Enough Evidence
    ▼
[Fact Verification Agent]
    │ Claim extraction and verification
    │ Confidence scoring with citations
    ▼
[Response Generation Agent]
    │ Generates corrected response with citations
    ▼
[Evaluation Agent]
    │ Computes metrics: Precision, Recall, F1, Accuracy,
    │ Hallucination Rate, Latency, Retrieval Quality
    ▼
Final Verified Response + Metrics
```

## Agent Interaction Diagram

```
                    ┌───────────────────┐
                    │   User Request    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Query Under-     │
                    │  standing Agent   │
                    └─────────┬─────────┘
                              │
                 ┌────────────▼────────────┐
                 │    Retrieval Agent      │
                 │  ┌──────────────────┐  │
                 │  │ FAISS VectorDB   │  │
                 │  │ PDF/Text/Web     │  │
                 │  │ Embeddings(BGE)  │  │
                 │  └──────────────────┘  │
                 └────────────┬────────────┘
                              │
                 ┌────────────▼────────────┐
                 │ Evidence Ranking Agent  │
                 │  ┌──────────────────┐  │
                 │  │ Cross-Encoder    │  │
                 │  │ Deduplication    │  │
                 │  │ Top-K Selection  │  │
                 │  └──────────────────┘  │
                 └────────────┬────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                                   │
  ┌─────────▼─────────┐              ┌─────────▼─────────┐
  │  Hallucination    │              │  Fact Verification │
  │  Detection Agent  │─────────────▶│      Agent         │
  │  (NLI Model)      │              │  (Claim Verify)    │
  └─────────┬─────────┘              └─────────┬──────────┘
            │                                   │
            └─────────────────┬─────────────────┘
                              │
                 ┌────────────▼────────────┐
                 │ Response Generation     │
                 │       Agent             │
                 │  (Verified + Citations) │
                 └────────────┬────────────┘
                              │
                 ┌────────────▼────────────┐
                 │   Evaluation Agent      │
                 │  (Metrics & Benchmark)  │
                 └─────────────────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestration | LangGraph | Multi-agent workflow management |
| LLM Framework | LangChain | Agent construction and chaining |
| Vector Database | FAISS | Similarity search for document retrieval |
| Embeddings | Sentence-Transformers (BGE/MiniLM/E5) | Document and query embeddings |
| Reranking | Cross-Encoder (ms-marco-MiniLM) | Evidence reranking |
| NLI Model | HuggingFace (DeBERTa-v3) | Natural Language Inference for hallucination detection |
| LLM (Local) | Ollama (Mistral/Phi/Llama/Gemma) | Text generation (optional) |
| LLM (Free) | HuggingFace Inference | Fallback text generation |
| Backend | FastAPI | REST API server |
| Frontend | Streamlit | Interactive web interface |
| Database | SQLite | Metadata and evaluation storage |
| Language | Python 3.10+ | Primary language |

## Folder Structure

```
hallucination_detector/
├── agents/
│   ├── __init__.py
│   ├── query_understanding_agent.py
│   ├── retrieval_agent.py
│   ├── evidence_ranking_agent.py
│   ├── hallucination_detection_agent.py
│   ├── fact_verification_agent.py
│   ├── response_generation_agent.py
│   └── evaluation_agent.py
├── core/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── chunking.py
│   ├── reranker.py
│   ├── nli_model.py
│   ├── llm_provider.py
│   └── database.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   └── dependencies.py
├── frontend/
│   ├── app.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── upload.py
│   │   ├── query.py
│   │   ├── results.py
│   │   └── metrics.py
│   └── utils.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── benchmark.py
│   └── datasets.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── vector_store/
├── tests/
│   ├── test_agents/
│   ├── test_core/
│   └── test_api/
├── docs/
├── scripts/
│   ├── setup.sh
│   └── ingest_documents.py
├── models/
├── orchestrator.py
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

## Installation

```bash
# Clone and navigate
cd hallucination_detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Ollama for local LLM
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull mistral

# Run the application
# Backend
uvicorn api.main:app --reload --port 8000

# Frontend
streamlit run frontend/app.py
```

## Example Input

```json
{
  "query": "What are the main causes of climate change?",
  "llm_response": "Climate change is primarily caused by volcanic eruptions and solar flares. Human activities have no significant impact on global warming.",
  "documents": ["climate_science.pdf", "ipcc_report.pdf"]
}
```

## Example Output

```json
{
  "verified_response": "Climate change is primarily caused by human activities, including burning fossil fuels, deforestation, and industrial processes that release greenhouse gases.",
  "hallucination_report": {
    "total_claims": 3,
    "supported": 0,
    "contradicted": 2,
    "not_enough_evidence": 1,
    "hallucination_rate": 0.67
  },
  "sentence_analysis": [
    {
      "sentence": "Climate change is primarily caused by volcanic eruptions and solar flares.",
      "label": "CONTRADICTED",
      "confidence": 0.94,
      "evidence": "The dominant cause of global warming is human activities..."
    },
    {
      "sentence": "Human activities have no significant impact on global warming.",
      "label": "CONTRADICTED", 
      "confidence": 0.97,
      "evidence": "Human influence has warmed the climate at a rate unprecedented..."
    }
  ],
  "metrics": {
    "precision": 0.92,
    "recall": 0.88,
    "f1_score": 0.90,
    "latency_ms": 2340
  }
}
```

## License

MIT License
