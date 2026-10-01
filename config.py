import os
from pathlib import Path

from dotenv import dotenv_values


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

_config = dotenv_values(ENV_FILE)


def get_api_key() -> str:
    api_key = _config.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Please check the .env file."
        )

    return api_key


def get_model_name() -> str:
    return _config.get("MODEL_NAME", "openai/gpt-oss-20b")