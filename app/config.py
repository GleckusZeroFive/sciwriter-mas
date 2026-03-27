"""Application configuration via pydantic-settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:8b"
    llm_api_key: str = "ollama"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "sciwriter_knowledge"

    # --- Embeddings ---
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = 1024

    # --- RAG ---
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_semantic_top_k: int = 20
    rag_bm25_top_k: int = 20
    rag_final_top_k: int = 5
    rag_score_threshold: float = 0.35

    # --- PostgreSQL ---
    pg_host: str = "localhost"
    pg_port: int = 5433
    pg_db: str = "sciwriter"
    pg_user: str = "sciwriter"
    pg_password: str = "sciwriter_pass"

    # --- Factory ---
    factory_collect_interval_reddit: int = 900       # 15 min
    factory_collect_interval_hn: int = 900           # 15 min
    factory_collect_interval_arxiv: int = 3600       # 1 hour
    factory_collect_interval_rss: int = 1800         # 30 min
    factory_generate_interval: int = 3000            # 50 min
    factory_publish_interval: int = 3600             # 1 hour
    factory_reddit_client_id: str = ""
    factory_reddit_client_secret: str = ""
    factory_reddit_user_agent: str = "SciWriter/1.0"

    # --- Preset ---
    article_preset: str = "habr"

    # --- Web Search ---
    web_search_max_results: int = 10

    # --- Workflow ---
    max_revisions: int = 2
    fact_check_pass_threshold: float = 7.0

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Streamlit ---
    streamlit_port: int = 8501

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def presets_dir(self) -> Path:
        return self.project_root / "app" / "presets"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def media_dir(self) -> Path:
        d = self.project_root / "output" / "media"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def pg_dsn(self) -> str:
        return f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"


settings = Settings()
