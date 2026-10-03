from functools import lru_cache
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone
from pydantic import BaseModel, Field

from app.config import settings
from app.models import get_embedder, get_llm

NOT_FOUND = "I couldn't find that in the Agentic AI eBook, so I can't answer it."

GENERATE_SYSTEM = f"""You are a question-answering assistant for ONE document: the Agentic AI eBook.
Rules:
1. Answer ONLY using the numbered context excerpts provided. Never use outside knowledge.
2. If the excerpts do not contain the answer, reply exactly: {NOT_FOUND}
3. Be concise. Cite the excerpts you used like [1], [2].
4. Do not speculate, and do not add facts that are not in the excerpts.
5. Do not add counts, step numbers, labels or structure (e.g. "three steps", "a four-part framework")
   unless the excerpts state them explicitly. Describe only what the excerpts say."""

REVISE_SYSTEM = """You fix a draft answer so that it is fully supported by the context.
Rewrite the draft: remove or rephrase every unsupported claim listed, keep the supported content and the
[n] citations, and add nothing new. If almost nothing is supported, reply exactly: """ + NOT_FOUND

VERIFY_SYSTEM = """You are a strict fact-checker. Given CONTEXT and an ANSWER, decide whether
every factual claim in the ANSWER is directly supported by the CONTEXT. Ignore citation markers
like [1]. If any claim is unsupported or goes beyond the context, set supported=false."""


class RAGState(TypedDict, total=False):
    question: str
    top_k: int
    contexts: list[dict]
    retrieval_score: float
    answer: str
    answered: bool
    grounded: bool
    confidence: float
    reason: str
    needs_revision: bool
    unsupported: list[str]
    revisions: int


class GroundingCheck(BaseModel):
    supported: bool = Field(description="True only if all claims in the answer are supported by the context")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims in the answer that the context does not support")
    reason: str = Field(description="One-sentence justification")


# ---------- lazy singletons ----------
@lru_cache(maxsize=1)
def _index():
    settings.validate()
    return Pinecone(api_key=settings.pinecone_api_key).Index(settings.index_name)


def _embedder():
    return get_embedder()


def _llm():
    return get_llm()


def _calibrate(score: float) -> float:
    """Map raw cosine similarity to a rough 0-1 confidence (0.2 -> 0, 0.7 -> 1). Heuristic; tune per corpus."""
    return round(max(0.0, min(1.0, (score - 0.2) / 0.5)), 3)


# ---------- nodes ----------
def retrieve(state: RAGState) -> RAGState:
    k = state.get("top_k") or settings.top_k
    vec = _embedder().embed_query(state["question"])
    res = _index().query(vector=vec, top_k=k, include_metadata=True, namespace=settings.namespace)
    contexts = [
        {"id": m.id, "text": m.metadata.get("text", ""), "page": int(m.metadata.get("page", 0)),
         "score": round(float(m.score), 4)}
        for m in res.matches
    ]
    top = [c["score"] for c in contexts[:3]]
    return {"contexts": contexts, "retrieval_score": sum(top) / len(top) if top else 0.0}


def route_after_retrieve(state: RAGState) -> str:
    contexts = state.get("contexts", [])
    if not contexts or contexts[0]["score"] < settings.min_score:
        return "refuse"
    return "generate"


def refuse(state: RAGState) -> RAGState:
    return {"answer": NOT_FOUND, "answered": False, "grounded": True,
            "confidence": _calibrate(state.get("retrieval_score", 0.0)),
            "reason": "No sufficiently relevant passage was retrieved."}


def _format_context(contexts: list[dict]) -> str:
    return "\n\n".join(f"[{i}] (page {c['page']}) {c['text']}" for i, c in enumerate(contexts, 1))


def generate(state: RAGState) -> RAGState:
    msg = _llm().invoke([
        SystemMessage(content=GENERATE_SYSTEM),
        HumanMessage(content=f"Context:\n{_format_context(state['contexts'])}\n\nQuestion: {state['question']}"),
    ])
    return {"answer": msg.content.strip()}


def verify(state: RAGState) -> RAGState:
    answer = state["answer"]
    conf = _calibrate(state["retrieval_score"])
    if answer.startswith(NOT_FOUND[:30]):  # model itself declined
        return {"answer": NOT_FOUND, "answered": False, "grounded": True, "confidence": conf,
                "needs_revision": False, "reason": "The model found no answer in the retrieved context."}
    check = _llm().with_structured_output(GroundingCheck).invoke([
        SystemMessage(content=VERIFY_SYSTEM),
        HumanMessage(content=f"CONTEXT:\n{_format_context(state['contexts'])}\n\nANSWER:\n{answer}"),
    ])
    if check.supported:
        reason = check.reason
        if state.get("revisions", 0):
            reason = "Passed after one revision removing unsupported claims. " + reason
        return {"answered": True, "grounded": True, "confidence": conf,
                "needs_revision": False, "reason": reason}
    if state.get("revisions", 0) < 1:  # try one repair before giving up
        return {"needs_revision": True, "unsupported": check.unsupported_claims or [check.reason]}
    return {"answer": NOT_FOUND, "answered": False, "grounded": False, "needs_revision": False,
            "confidence": round(conf * 0.5, 3),
            "reason": f"Draft answer failed grounding check: {check.reason}"}


def revise(state: RAGState) -> RAGState:
    problems = "\n".join(f"- {c}" for c in state.get("unsupported", []))
    msg = _llm().invoke([
        SystemMessage(content=REVISE_SYSTEM),
        HumanMessage(content=(f"Context:\n{_format_context(state['contexts'])}\n\n"
                              f"Question: {state['question']}\n\nDraft answer:\n{state['answer']}\n\n"
                              f"Unsupported claims to remove or fix:\n{problems}")),
    ])
    return {"answer": msg.content.strip(), "revisions": state.get("revisions", 0) + 1,
            "needs_revision": False}


def route_after_verify(state: RAGState) -> str:
    return "revise" if state.get("needs_revision") else "done"


# ---------- graph ----------
@lru_cache(maxsize=1)
def get_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("refuse", refuse)
    g.add_node("revise", revise)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"revise": "revise", "done": END})
    g.add_edge("revise", "verify")
    g.add_edge("refuse", END)
    return g.compile()