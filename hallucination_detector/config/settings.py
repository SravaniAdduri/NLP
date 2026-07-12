import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for all ML models used in the system."""
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    nli_model: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    generation_model: str = os.getenv("GENERATION_MODEL", "microsoft/Phi-3-mini-4k-instruct")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral")


@dataclass
class VectorStoreConfig:
    """Configuration for FAISS vector store."""
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "5"))


@dataclass
class DatabaseConfig:
    """Configuration for database."""
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/hallucination_detector.db")


@dataclass
class APIConfig:
    """Configuration for FastAPI server."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))


@dataclass
class AppConfig:
    """Main application configuration aggregating all sub-configs."""
    models: ModelConfig = field(default_factory=ModelConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"


def get_config() -> AppConfig:
    """Factory function to create application configuration."""
    return AppConfig()
