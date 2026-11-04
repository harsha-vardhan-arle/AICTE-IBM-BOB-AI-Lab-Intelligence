"""
config.py — Centralised configuration loader for the AI Lab Manual Generator.
Reads all settings from environment variables (via python-dotenv).
"""

import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()


class Config:
    """All application-wide configuration constants."""

    # ── IBM watsonx.ai ───────────────────────────────────────
    IBM_CLOUD_API_KEY: str = os.getenv("IBM_CLOUD_API_KEY", "")
    IBM_CLOUD_URL: str = os.getenv(
        "IBM_CLOUD_URL", "https://us-south.ml.cloud.ibm.com"
    )
    IBM_WATSONX_PROJECT_ID: str = os.getenv("IBM_WATSONX_PROJECT_ID", "")

    # ── Granite Model IDs ────────────────────────────────────
    GRANITE_LLM_MODEL_ID: str = os.getenv(
        "GRANITE_LLM_MODEL_ID", "ibm/granite-4-h-small"
    )
    GRANITE_EMBEDDING_MODEL_ID: str = os.getenv(
        "GRANITE_EMBEDDING_MODEL_ID", "ibm/granite-embedding-278m-multilingual"
    )

    # ── IBM Cloud Object Storage ─────────────────────────────
    IBM_COS_API_KEY: str = os.getenv("IBM_COS_API_KEY", "")
    IBM_COS_INSTANCE_CRN: str = os.getenv("IBM_COS_INSTANCE_CRN", "")
    IBM_COS_ENDPOINT: str = os.getenv(
        "IBM_COS_ENDPOINT",
        "https://s3.us-south.cloud-object-storage.appdomain.cloud",
    )
    IBM_COS_BUCKET_NAME: str = os.getenv("IBM_COS_BUCKET_NAME", "lab-manual-corpus")

    # ── ChromaDB Vector Store ────────────────────────────────
    CHROMA_PERSIST_DIRECTORY: str = os.getenv(
        "CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"
    )
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "lab_curriculum"
    )

    # ── Flask ─────────────────────────────────────────────────
    FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    # ── RAG ───────────────────────────────────────────────────
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
    RAG_TOP_K_RESULTS: int = int(os.getenv("RAG_TOP_K_RESULTS", "5"))
    RAG_SIMILARITY_THRESHOLD: float = float(
        os.getenv("RAG_SIMILARITY_THRESHOLD", "0.65")
    )

    # ── LLM Generation ───────────────────────────────────────
    LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "1024"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))
    LLM_TOP_K: int = int(os.getenv("LLM_TOP_K", "50"))
    LLM_REPETITION_PENALTY: float = float(
        os.getenv("LLM_REPETITION_PENALTY", "1.1")
    )

    # ── Data Directories ─────────────────────────────────────
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    CORPUS_DIR: str = os.getenv("CORPUS_DIR", "./data/corpus")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./data/output")
    UPLOADS_DIR: str = os.getenv("UPLOADS_DIR", "./data/uploads")

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of missing required configuration keys."""
        missing = []
        required = [
            ("IBM_CLOUD_API_KEY", cls.IBM_CLOUD_API_KEY),
            ("IBM_WATSONX_PROJECT_ID", cls.IBM_WATSONX_PROJECT_ID),
        ]
        for name, value in required:
            if not value or value.startswith("your_"):
                missing.append(name)
        return missing

    @classmethod
    def ensure_directories(cls) -> None:
        """Create all required data directories if they don't exist."""
        for directory in [
            cls.DATA_DIR,
            cls.CORPUS_DIR,
            cls.OUTPUT_DIR,
            cls.UPLOADS_DIR,
            cls.CHROMA_PERSIST_DIRECTORY,
        ]:
            os.makedirs(directory, exist_ok=True)
