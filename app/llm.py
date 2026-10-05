from typing import Optional
from langchain_openai import ChatOpenAI
from app.config import settings


def get_llm(temperature: float = 0.0, model: Optional[str] = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.ollama_model,
        temperature=temperature,
        base_url=settings.ollama_base_url,
        api_key="ollama",
    )


def get_eval_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.ollama_model,
        temperature=0.0,
        base_url=settings.ollama_base_url,
        api_key="ollama",
    )