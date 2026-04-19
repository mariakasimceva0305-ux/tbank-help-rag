import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.rag.graph import app as rag_app


def _to_langchain_messages(chat_history: list[dict]) -> list[HumanMessage | AIMessage]:
    messages: list[HumanMessage | AIMessage] = []
    for message in chat_history:
        message_cls = HumanMessage if message["role"] == "user" else AIMessage
        messages.append(message_cls(content=message["content"]))
    return messages


def _render_sources(sources: list[str]) -> None:
    if not sources:
        return

    with st.expander("Источники"):
        for source in sources:
            st.markdown(f"- {source}")

st.title("Т-Банк Ассистент")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        _render_sources(message.get("sources", []))

if prompt := st.chat_input("Задайте вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ищу ответ..."):
            result = rag_app.invoke(
                {"messages": _to_langchain_messages(st.session_state.messages)},
                config={"configurable": {"thread_id": "default_user_session"}}
            )
            answer = result["messages"][-1].content
            sources = result.get("sources", [])
            st.write(answer)
            _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
