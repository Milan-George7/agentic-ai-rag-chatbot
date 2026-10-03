# final
from functools import lru_cache

from app.config import settings 


@lru_cache(maxsize=1)
def get_embedder():
    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings  # runs locally, no API key
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


@lru_cache(maxsize=1)
def get_llm():
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=settings.llm_model, temperature=0, api_key=settings.groq_api_key)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)
