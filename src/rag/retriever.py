from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from dotenv import load_dotenv

from src.rag.config import COLLECTION_NAME, get_embeddings, get_qdrant_client, get_sparse_embeddings

load_dotenv(override=False)

FINAL_K = int(os.getenv("RAG_FINAL_K", "6"))


@lru_cache(maxsize=1)
def get_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
    )


def retrieve(query: str) -> list[Document]:
    normalized_query = query.strip()
    if not normalized_query:
        return []

    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' does not exist. Run indexing first."
        )

    return get_store().similarity_search(normalized_query, k=FINAL_K)


if __name__ == "__main__":
    results = retrieve("как открыть вклад?")
    for doc in results:
        print(doc.page_content)
