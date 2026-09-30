# Agentic AI eBook Chatbot

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Pinecone](https://img.shields.io/badge/Pinecone-vector%20DB-000000)](https://www.pinecone.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-LLM-F55036)](https://console.groq.com)

**[Quick start](#quick-start)** · **[How it works](#how-it-works)** · **[API response](#what-the-api-returns)** · **[Sample queries](#sample-queries)** · **[Settings](#settings)** · **[Troubleshooting](#troubleshooting)**

A chatbot that answers questions about the [Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf) and nothing else.
Ask it something the eBook doesn't cover and it says so instead of guessing.
Every answer comes back with the passages it used and a confidence score.

- Built with **LangGraph** (pipeline), **Pinecone** (vector search) and **FastAPI** (chat API)
- Optional **Streamlit** chat window
- Works with **Groq + free local embeddings** (the setup I tested) or with **OpenAI**

---

## Quick start

You need Python 3.10+, a Pinecone API key, and either a Groq key or an OpenAI key.

**1. Install**
- Windows: `py -3.13 -m venv .venv`, then `.venv\Scripts\activate`
- macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
- Then: `python -m pip install -r requirements.txt`
- Groq setup only: `python -m pip install -r requirements-groq.txt` (large download, it includes PyTorch)

**2. Add your keys**
- Copy `.env.example` to `.env` (`copy` on Windows, `cp` elsewhere) and edit it
- Groq + local embeddings:
  ```
  GROQ_API_KEY=gsk_...
  PINECONE_API_KEY=pcsk_...
  EMBEDDING_PROVIDER=huggingface
  LLM_PROVIDER=groq
  LLM_MODEL=openai/gpt-oss-120b
  PINECONE_INDEX=agentic-ai-ebook-384
  ```
- OpenAI instead:
  ```
  OPENAI_API_KEY=sk-...
  PINECONE_API_KEY=pcsk_...
  ```
- No quotes, no spaces around `=`

**3. Load the eBook into Pinecone (one time)**
```
python -m app.ingest
```

**4. Start the API**
```
uvicorn app.api:app --port 8000
```
Then open http://127.0.0.1:8000/docs, click `POST /chat`, then **Try it out**.

**5. Optional chat window** (second terminal, same folder, venv active)
```
streamlit run ui.py
```

---

## How it works

**Loading the eBook** (`app/ingest.py`)
- Downloads the PDF and reads the text page by page
- Cuts it into ~1000-character chunks with 150 characters of overlap, and drops tiny fragments like headings and page numbers
- Turns each chunk into an embedding and stores it in Pinecone along with its text and page number

**Answering a question** (`app/rag_graph.py`, a small LangGraph)
- **Retrieve:** embed the question and pull the closest chunks from Pinecone
- **Gate:** if even the best chunk is a weak match (below `MIN_SCORE`), skip the LLM and say the eBook doesn't cover it
- **Generate:** the LLM writes an answer from those chunks only, with `[1]`, `[2]` style citations
- **Verify:** a second LLM call checks every claim against the chunks. If something isn't supported, the answer is replaced with a refusal

**Why the answers stay on topic**
- The prompt allows only the retrieved passages and gives the model a fixed sentence to use when the answer isn't there
- Temperature is 0, so the model doesn't get creative
- The verify step catches drafts that drift beyond the source

**About the confidence score**
- It comes from how closely the top 3 chunks matched the question, rescaled to 0–1, and it's halved if verification fails
- Treat it as a rough relevance signal, not a probability
- With the local embeddings, good matches often land near 1.0, so look at `retrieval_score` and the per-chunk scores when you want finer detail

**Where things live**
- `app/config.py`: settings from `.env`
- `app/models.py`: picks the embedding model and LLM
- `app/ingest.py`: loading the eBook
- `app/rag_graph.py`: the pipeline
- `app/api.py`: the FastAPI endpoints
- `ui.py`: the Streamlit window
- `scripts/run_samples.py`: runs the sample queries

---

## What the API returns

`POST /chat` with `{"question": "...", "top_k": 5}` gives:

```json
{
  "answer": "… [1][3]",
  "answered": true,
  "grounded": true,
  "confidence": 0.9,
  "retrieval_score": 0.7981,
  "reason": "Why the verifier accepted or rejected the answer",
  "contexts": [
    {"id": "p7-c0", "page": 7, "score": 0.7816, "text": "…"}
  ]
}
```

- `answer`: the final answer, or a polite refusal
- `answered`: `false` when the eBook doesn't cover the question
- `grounded`: `false` when the draft failed the verify step
- `confidence` and `retrieval_score`: see the notes above
- `contexts`: the chunks used, with page numbers and similarity scores

Off-topic questions get: *"I couldn't find that in the Agentic AI eBook, so I can't answer it."*

---

## Sample queries

1. How does Agentic AI differ from generative AI and traditional AI?
2. What are perception, reasoning, planning, learning and execution in an agentic AI system?
3. How are multi-agent systems structured and why are they used?
4. What practical applications of Agentic AI does the eBook cover?
5. How can an organization assess its readiness for Agentic AI?
6. What is the capital of France? *(should be refused)*

More detail is in [`docs/sample_queries.md`](docs/sample_queries.md). To run all six and save the results:
```
python -m scripts.run_samples
```
They're written to `docs/sample_outputs.md`. On Groq's free plan you may hit a rate limit; wait a minute and run it again.

---

## Settings

Set these in `.env`. Defaults are fine to start with.

- `EMBEDDING_PROVIDER`: `openai` or `huggingface`
- `LLM_PROVIDER`: `openai` or `groq`
- `LLM_MODEL`: needs tool-calling support, because the verify step uses structured output
- `PINECONE_INDEX`: name it after the embedding size, since an index is locked to one size (384 for MiniLM, 1536 for OpenAI)
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: 1000 / 150. Re-run ingestion after changing them
- `TOP_K`: how many chunks to retrieve (5)
- `MIN_SCORE`: the refusal threshold (0.25). Ask an off-topic question, look at its `retrieval_score`, and set this just above it

---

## Troubleshooting

- **`pip` launcher error on Windows:** use `python -m pip ...` inside a fresh venv
- **`python` not found but `py` works:** that's the Microsoft Store shortcut; create the venv with `py -3.x -m venv .venv`
- **Pinecone 401:** your `.env` still has placeholder keys; check the length with `python -c "from app.config import settings as s; print(len(s.pinecone_api_key))"`
- **OpenAI 401 mentioning a `gsk_` key:** that's a Groq key; put it in `GROQ_API_KEY` and set `LLM_PROVIDER=groq`
- **`.env` changes seem ignored:** a Windows environment variable with the same name wins over `.env` (check with `set NAME`), and the API only reads `.env` at startup, so restart uvicorn
- **Hugging Face error about `text-embedding-3-small`:** delete the leftover `EMBEDDING_MODEL` line from `.env`
- **Index dimension mismatch:** delete the index in the Pinecone console or pick a new `PINECONE_INDEX` name
- **Groq `model_not_found`:** the model was retired or isn't on your account. List what's available and set `LLM_MODEL`:
  ```
  python -c "import requests; from app.config import settings as s; r=requests.get('https://api.groq.com/openai/v1/models', headers={'Authorization':'Bearer '+s.groq_api_key}); print(sorted(m['id'] for m in r.json()['data']))"
  ```
- **Rate limit (429):** Groq's free plan is tight per minute, and each question makes two LLM calls. Wait a bit, lower `top_k`, or upgrade

---

## Good to know

- Very short questions like "What is Agentic AI?" can match headings and section intros; more specific questions work better
- The cover page has no extractable text, and a passage that spans a page break gets split
- Each answered question costs two LLM calls (answer plus verification)
- There's no chat memory; every question stands alone
- Tested end to end with Groq (`openai/gpt-oss-120b`) and local MiniLM embeddings. The OpenAI path is implemented but hasn't been run
