# Ассистент по справке Т-Банка

Чат по публичным страницам справки: загрузка через sitemap, гибридный поиск в Qdrant, формирование ответа через LangGraph и Groq. Интерфейс — Streamlit; рядом с ответом показаны ссылки на источники.

## Как устроен поиск

Документы разбиваются на фрагменты и индексируются с BAAI/bge-m3 и BM25. Перед поиском запрос уточняется с учётом истории диалога. Найденные фрагменты передаются LLM как контекст. Модели на документах банка не дообучаются.

[Демонстрация](docs/demo/demo.mp4) · [Презентация](docs/presentation/slides.pdf)

## Запуск

Python 3.12, Docker Compose для Qdrant, ключ Groq. Первый запуск требует доступа к моделям Hugging Face и сайту справки.

```bash
pip install -r requirements.txt
cp .env.example .env
# Укажите GROQ_API_KEY в .env

docker compose up -d qdrant
python -m src.rag.parse_sitemap
python -m src.rag.run_indexing
streamlit run app.py
```

В PowerShell вместо `cp` можно использовать `Copy-Item`. Для небольшой первичной индексации задайте `RAG_MAX_PAGES=20`. Файловый режим Qdrant включается через `QDRANT_PATH`; при нём контейнер не нужен.

После индексации приложение и Qdrant можно запускать вместе: `docker compose up --build`. Интерфейс доступен на порту 8501.

## Конфигурация и код

Переменные перечислены в `.env.example`: `GROQ_MODEL`, `EMBEDDING_MODEL`, `QDRANT_COLLECTION`, `QDRANT_HOST`, `QDRANT_PORT`, `RAG_FINAL_K`.

- `src/rag/parse_sitemap.py`, `indexer.py`, `run_indexing.py` — сбор и индексация справки;
- `src/rag/retriever.py` — поиск;
- `src/rag/graph.py` — диалог и формирование ответа;
- `src/rag/debug_retrieval.py` — проверка найденных фрагментов;
- `app.py` — интерфейс.

Качество зависит от актуальности проиндексированной справки. Отдельной размеченной выборки для численной оценки ответов пока нет. Это самостоятельный проект на публичных материалах, не официальный сервис поддержки банка.
