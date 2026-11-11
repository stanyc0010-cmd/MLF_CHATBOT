from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLF_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://mlf:mlf@localhost:5432/mlf_assistant"
    test_database_url: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 1
    speech_transcription_model: str = "gpt-4o-mini-transcribe"
    speech_synthesis_model: str = "gpt-4o-mini-tts"
    speech_voice: str = "alloy"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = ""
    admin_username: str = "admin"
    admin_password: str = ""
    admin_password_hash: str = ""
    allowed_origins: str = ""
    max_upload_mb: int = 40
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    min_relevance_score: float = 0.35
    retrieval_limit: int = 12
    llm_input_cost_per_million: float | None = None
    llm_output_cost_per_million: float | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()