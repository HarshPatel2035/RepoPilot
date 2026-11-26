from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://repopilot:repopilot@localhost:5432/repopilot"
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    # embed_dim must match the embedding model's output size (bge-small = 384).
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    top_k: int = 8
    search_mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    rerank: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # A different model family from llm_model, so the judge is not grading its own answers.
    judge_model: str = "openai/gpt-oss-120b"
    agent_max_tool_rounds: int = 6
    # Optional: raises GitHub's unauthenticated limit of 60 issue requests per hour.
    github_token: str = ""
    max_file_bytes: int = 200_000
    max_repo_files: int = 5_000
