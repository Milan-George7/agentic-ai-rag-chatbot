"""Ingestion: download PDF -> extract text per page -> chunk -> embed -> upsert to Pinecone.

Usage:
    python -m app.ingest            # ingest (idempotent: same chunk IDs are overwritten)
    python -m app.ingest --reset    # wipe the namespace first
"""
import argparse
import io
import re
import time

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

from app.config import settings
from app.models import get_embedder


def download_pdf(url: str) -> bytes:
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (rag-ingest)"})
    resp.raise_for_status()
    return resp.content


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)          # collapse spaces
        text = re.sub(r"\n{3,}", "\n\n", text).strip()  # collapse blank lines
        if text:
            pages.append((i, text))
        else:
            print(f"  ! page {i}: no extractable text (image-only page?)")
    return pages


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page_no, text in pages:
        for j, piece in enumerate(splitter.split_text(text)):
            if len(piece.strip()) < 150: # skip page numbers / stray headers
                continue
            chunks.append({"id": f"p{page_no}-c{j}", "text": piece.strip(), "page": page_no})
    return chunks


def ensure_index(pc: Pinecone):
    if settings.index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{settings.index_name}' ...")
        pc.create_index(
            name=settings.index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
        )
        while not pc.describe_index(settings.index_name).status["ready"]:
            time.sleep(2)
    dim = pc.describe_index(settings.index_name).dimension
    if dim != settings.embedding_dim:
        raise SystemExit(
            f"Index '{settings.index_name}' has dimension {dim} but your embedding model produces "
            f"{settings.embedding_dim}. Delete the index in the Pinecone console or set a new PINECONE_INDEX name."
        )
    return pc.Index(settings.index_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete existing vectors in the namespace first")
    args = parser.parse_args()
    settings.validate()

    print(f"Downloading {settings.pdf_url}")
    pages = extract_pages(download_pdf(settings.pdf_url))
    chunks = chunk_pages(pages)
    if not chunks:
        raise SystemExit("No text extracted. The PDF may be scanned; add an OCR step (e.g. pytesseract).")
    print(f"Extracted {len(pages)} pages -> {len(chunks)} chunks")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = ensure_index(pc)
    if args.reset:
        try:
            index.delete(delete_all=True, namespace=settings.namespace)
        except Exception as e:  # namespace may not exist yet
            print(f"  (reset skipped: {e})")

    embedder = get_embedder()
    batch = 100
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        vectors = embedder.embed_documents([c["text"] for c in part])
        index.upsert(
            vectors=[
                {"id": c["id"], "values": v,
                 "metadata": {"text": c["text"], "page": c["page"], "source": settings.pdf_url}}
                for c, v in zip(part, vectors)
            ],
            namespace=settings.namespace,
        )
        print(f"  upserted {start + len(part)}/{len(chunks)}")
    print("Done.")


if __name__ == "__main__":
    main()
