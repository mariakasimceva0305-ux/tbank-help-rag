from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse
from qdrant_client import QdrantClient

load_dotenv(override=False)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "tbank_faq")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
# Если задан путь — Qdrant в embedded-режиме (файлы на диске), Docker не нужен.
QDRANT_PATH = os.getenv("QDRANT_PATH", "").strip()


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name="Qdrant/bm25")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if QDRANT_PATH:
        return QdrantClient(path=QDRANT_PATH)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_qdrant_url() -> str:
    return f"http://{QDRANT_HOST}:{QDRANT_PORT}"
