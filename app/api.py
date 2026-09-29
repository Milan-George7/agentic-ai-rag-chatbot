from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_graph import get_graph

app = FastAPI(title="Agentic AI eBook RAG Chatbot", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=10)


class ContextChunk(BaseModel):
    id: str
    page: int
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    answered: bool = Field(description="False when the bot declined because the eBook doesn't cover it")
    grounded: bool
    confidence: float = Field(description="Heuristic 0-1 score derived from retrieval similarity")
    retrieval_score: float = Field(description="Mean raw cosine score of the top-3 chunks")
    reason: str
    contexts: list[ContextChunk]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        out = get_graph().invoke({"question": req.question.strip(), "top_k": req.top_k})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    return ChatResponse(
        answer=out["answer"],
        answered=out["answered"],
        grounded=out["grounded"],
        confidence=out["confidence"],
        retrieval_score=round(out.get("retrieval_score", 0.0), 4),
        reason=out.get("reason", ""),
        contexts=out.get("contexts", []),
    )
