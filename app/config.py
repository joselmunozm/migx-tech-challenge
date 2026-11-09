"""Settings loaded from the environment and the local .env file."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    api_key: str = "dev-local-key"
    api_base_url: str = "http://127.0.0.1:8000"
    data_dir: Path = PROJECT_ROOT / "data"

    cohere_api_key: str = ""
    cohere_embed_model: str = "embed-v4.0"
    embed_dimension: int = 1024

    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemma-4-31b-it:free"
    openrouter_fallback_model: str = "nvidia/nemotron-3.5-super-vl:free"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    top_k: int = 3
    chunk_size: int = 300
    chunk_overlap: int = 50
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key.strip() and self.langfuse_secret_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
