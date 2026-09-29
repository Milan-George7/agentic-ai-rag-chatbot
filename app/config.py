import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

_EMB_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
_DEFAULT_EMB = {"openai": "text-embedding-3-small",
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2"}.get(_EMB_PROVIDER, "text-embedding-3-small")
_DEFAULT_DIM = "384" if _EMB_PROVIDER == "huggingface" else "1536"
_DEFAULT_LLM = "llama-3.3-70b-versatile" if _LLM_PROVIDER == "groq" else "gpt-4o-mini"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")

    embedding_provider: str = _EMB_PROVIDER   # openai | huggingface
    llm_provider: str = _LLM_PROVIDER         # openai | groq

    pdf_url: str = os.getenv("PDF_URL", "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf")
    index_name: str = os.getenv("PINECONE_INDEX", "agentic-ai-ebook")
    namespace: str = os.getenv("PINECONE_NAMESPACE", "ebook")
    cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    region: str = os.getenv("PINECONE_REGION", "us-east-1")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", _DEFAULT_EMB)
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", _DEFAULT_DIM))
    llm_model: str = os.getenv("LLM_MODEL", _DEFAULT_LLM)

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.25"))

    def validate(self) -> None:
        required = {"PINECONE_API_KEY": self.pinecone_api_key}
        if self.embedding_provider == "openai" or self.llm_provider == "openai":
            required["OPENAI_API_KEY"] = self.openai_api_key
        if self.llm_provider == "groq":
            required["GROQ_API_KEY"] = self.groq_api_key
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)} (see .env.example)")


settings = Settings()
