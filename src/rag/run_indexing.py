import os
from pathlib import Path

from src.rag.indexer import indexer, load_documents, split_docs


def main() -> None:
    urls_path = Path(__file__).with_name("urls.txt")
    urls = [
        line.strip()
        for line in urls_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    max_pages = os.getenv("RAG_MAX_PAGES", "").strip()
    if max_pages:
        urls = urls[: int(max_pages)]

    docs = load_documents(urls)
    all_chunks = split_docs(docs)

    indexer(all_chunks, recreate=True)
    print(f"Indexed {len(all_chunks)} chunks from {len(docs)} successful pages out of {len(urls)} urls.")


if __name__ == "__main__":
    main()
