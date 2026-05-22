"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Immutable application settings sourced from environment."""

    openai_api_key: str = ""
    oss_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    frontier_model_name: str = "gpt-4.1-mini"
    max_conversation_turns: int = 10

    model_config = {"env_file": ".env", "frozen": True}


def load_settings() -> Settings:
    """Load and return frozen application settings."""
    return Settings()
