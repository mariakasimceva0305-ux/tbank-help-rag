from __future__ import annotations

from src.rag.retriever import get_store


TEST_QUERIES = [
    "как открыть вклад",
    "как платить налог самозанятому",
    "как получить выписку",
]


def main() -> None:
    store = get_store()
    for query in TEST_QUERIES:
        print(f"\n=== QUERY: {query} ===")
        results = store.similarity_search_with_score(query, k=5)
        for index, (doc, score) in enumerate(results, start=1):
            source = (doc.metadata or {}).get("source", "unknown")
            preview = doc.page_content[:300].replace("\n", " ")
            print(f"{index}. score={score:.4f} source={source}")
            print(f"   {preview}\n")


if __name__ == "__main__":
    main()
