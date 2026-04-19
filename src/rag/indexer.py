from __future__ import annotations

import os
import re
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.config import (
    COLLECTION_NAME,
    QDRANT_PATH,
    get_embeddings,
    get_qdrant_url,
    get_sparse_embeddings,
)

os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
)

DROP_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "img",
    "picture",
    "video",
    "audio",
    "iframe",
    "form",
    "button",
    "header",
    "footer",
    "nav",
    "aside",
)
MIN_CONTENT_CHARS = 350

_session = requests.Session()
_session.headers.update({"User-Agent": os.environ["USER_AGENT"]})


def _pick_root(soup: BeautifulSoup):
    for selector in ("main", "article", "[role='main']", "body"):
        node = soup.select_one(selector)
        if node is not None:
            return node
    return soup


def _cleanup_node(root) -> None:
    for selector in DROP_SELECTORS:
        for node in root.select(selector):
            node.decompose()

    for node in root.select("[aria-hidden='true']"):
        node.decompose()


NOISE_LINES = {
    "Пропустить навигацию",
    "Личный кабинет",
    "На этой странице",
}


def _normalize_lines(strings: Iterable[str]) -> list[str]:
    lines: list[str] = []
    for raw in strings:
        text = re.sub(r"\s+", " ", raw).strip()
        if len(text) < 2 or text in NOISE_LINES:
            continue
        if lines and text == lines[-1]:
            continue
        lines.append(text)
    return lines


def load_documents(urls: list[str]) -> list[Document]:
    docs: list[Document] = []
    for url in urls:
        try:
            response = _session.get(url, timeout=30)
            response.raise_for_status()
            # Страницы Т-Банка в UTF-8; фиксируем кодировку до парсинга,
            # иначе в чанки может попасть mojibake.
            response.encoding = "utf-8"
        except Exception:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        root = _pick_root(soup)
        _cleanup_node(root)

        title = " ".join(soup.title.get_text(" ", strip=True).split()) if soup.title else url
        lines = _normalize_lines(root.stripped_strings)
        text = "\n".join(lines)
        if len(text) < MIN_CONTENT_CHARS:
            continue

        docs.append(
            Document(
                page_content=f"{title}\n\n{text}",
                metadata={
                    "source": url,
                    "title": title,
                    "content_length": len(text),
                },
            )
        )

    return docs


def split_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata = {**chunk.metadata, "chunk_id": chunk_id}
    return chunks


def indexer(chunks: list[Document], recreate: bool = True) -> QdrantVectorStore:
    # В embedded-режиме нельзя открывать два клиента к одному path одновременно.
    # Поэтому пересоздание коллекции делаем через force_recreate внутри from_documents.
    connect = {"path": QDRANT_PATH} if QDRANT_PATH else {"url": get_qdrant_url()}
    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name=COLLECTION_NAME,
        force_recreate=recreate,
        **connect,
    )
