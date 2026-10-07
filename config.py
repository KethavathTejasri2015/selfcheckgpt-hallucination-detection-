import os

from dotenv import load_dotenv


load_dotenv()


def get_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Please create a .env file."
        )

    return api_key


def get_model_name() -> str:
    return os.getenv(
        "MODEL_NAME",
        "llama-3.3-70b-versatile"
    )