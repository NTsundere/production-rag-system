from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings


def get_llm(temperature: float = 0.0, model: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Фабрика LLM на базе Google Gemini (REST)."""
    return ChatGoogleGenerativeAI(
        model=model or settings.gemini_model,
        temperature=temperature,
        google_api_key=settings.google_api_key,
        transport="rest",
    )


def get_eval_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.0,
        google_api_key=settings.google_api_key,
        transport="rest",
    )