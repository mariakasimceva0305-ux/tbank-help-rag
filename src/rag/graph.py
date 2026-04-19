from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.rag.retriever import retrieve
from src.state import State

load_dotenv(override=False)

NO_CONTEXT_ANSWER = (
    "К сожалению, у меня нет точной информации по этому вопросу. "
    "Рекомендую обратиться в поддержку Т-Банка по номеру 8-800-555-777-8"
)
SERVICE_UNAVAILABLE_ANSWER = (
    "Сейчас я не могу надежно получить данные из базы знаний. "
    "Рекомендую повторить запрос позже или обратиться в поддержку Т-Банка "
    "по номеру 8-800-555-777-8"
)


graph = StateGraph(State)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not configured")

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        streaming=True,
    )


def _last_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.type == "human":
            return message.content
    return messages[-1].content if messages else ""


def _rewrite_query(messages: list[BaseMessage]) -> str:
    user_query = _last_user_message(messages)
    if len(messages) <= 1:
        return user_query
        
    prompt = SystemMessage(
        content="Переформулируй последний вопрос пользователя с учетом истории диалога так, "
        "чтобы он стал самостоятельным запросом для поиска в базе знаний. "
        "Если вопрос не требует контекста (например, это приветствие или уже полный вопрос), "
        "верни его как есть. Напиши ТОЛЬКО переформулированный запрос без кавычек."
    )
    
    try:
        messages_to_pass = [prompt] + messages[-4:]
        response = get_llm().invoke(messages_to_pass)
        return response.content.strip()
    except Exception:
        return user_query


def _format_context(documents: list) -> str:
    context_blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source = metadata.get("source") or metadata.get("url") or "Источник не указан"
        title = metadata.get("title") or "Без заголовка"
        context_blocks.append(
            f"[{index}]\nИсточник: {source}\nЗаголовок: {title}\nФрагмент:\n{document.page_content}"
        )
    return "\n\n".join(context_blocks)


def _extract_sources(documents: list) -> list[str]:
    sources: list[str] = []
    for document in documents:
        source = (document.metadata or {}).get("source") or (document.metadata or {}).get("url")
        if source and source not in sources:
            sources.append(source)
    return sources


def retrieve_node(state: State):
    query = _rewrite_query(state["messages"])
    try:
        docs = retrieve(query)
    except Exception as exc:
        return {"documents": [], "sources": [], "error": str(exc)}

    return {"documents": docs, "sources": _extract_sources(docs), "error": None}


def generate_node(state: State):
    if state.get("error"):
        return {"messages": [AIMessage(content=SERVICE_UNAVAILABLE_ANSWER)]}

    docs = state.get("documents", [])
    docs_text = _format_context(docs) if docs else "Нет найденной информации."
    
    system_prompt = f"""Ты — виртуальный ассистент поддержки Т-Банка. Твоя задача — профессионально и точно отвечать на вопросы клиентов.

ПРАВИЛА ОТВЕТА:
1. Используй факты ТОЛЬКО из блока <context>.
2. Отвечай на русском языке, будь максимально вежлив, обращайся к клиенту строго на "Вы".
3. Если клиент просто здоровается, прощается или благодарит (small talk), ответь вежливо от себя.
4. Если клиент задает вопрос по банку, а в блоке <context> нет точной информации для ответа, ответь СТРОГО по шаблону:
"{NO_CONTEXT_ANSWER}"
5. Не выдумывай тарифы, сроки, комиссии, статусы услуг и внутренние процессы.
6. Если информации достаточно, сначала дай прямой ответ на вопрос, затем при необходимости добавь 2-4 короткие полезные детали.

<context>
{docs_text}
</context>
"""

    messages_to_pass = [SystemMessage(content=system_prompt)] + state["messages"][-5:]

    try:
        answer = get_llm().invoke(messages_to_pass)
    except Exception:
        return {"messages": [AIMessage(content=SERVICE_UNAVAILABLE_ANSWER)]}

    return {"messages": [answer]}


graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)


# Добавляем MemorySaver для сохранения истории диалогов
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    # config определяет "сессию" (или ID пользователя), чтобы граф мог восстанавливать контекст
    config = {"configurable": {"thread_id": "user_123"}}

    print("Клиент: Привет! Меня зовут Артём.")
    print("Бот: ", end="", flush=True)
    
    # Пример 1: Отправляем сообщение и печатаем ответ в потоковом режиме (Streaming)
    for msg, metadata in app.stream(
        {"messages": [HumanMessage(content="Привет! Меня зовут Артём.")]},
        config=config,
        stream_mode="messages",
    ):
        if msg.type == "ai" and msg.content:
            print(msg.content, end="", flush=True)
    print("\n")

    # Пример 2: Проверяем, что бот запомнил имя (Память работает)
    print("Клиент: Как меня зовут?")
    print("Бот: ", end="", flush=True)
    for msg, metadata in app.stream(
        {"messages": [HumanMessage(content="Как меня зовут?")]},
        config=config,
        stream_mode="messages",
    ):
        if msg.type == "ai" and msg.content:
            print(msg.content, end="", flush=True)
    print("\n")
