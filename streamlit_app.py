from __future__ import annotations

import asyncio
import importlib
import os
from io import BytesIO
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv(override=True)

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🧠",
    layout="wide",
)

EXAMPLES = [
    "Résume le document",
    "Quels sont les points clés ?",
    "De quoi parle ce PDF ?",
]


def serialize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


def to_langchain_messages(history: list[dict[str, str]]) -> list:
    converted = []
    for message in history:
        if message["role"] == "user":
            converted.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            converted.append(AIMessage(content=message["content"]))
    return converted


def extract_answer(messages: list) -> str:
    answer = ""
    for message in messages:
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            answer = serialize_content(message.content)
    return answer


@st.cache_resource(show_spinner="Chargement de l'agent RAG…")
def load_rag(api_key: str, _version: int = 6):
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquante.")
    os.environ["OPENAI_API_KEY"] = api_key
    import agentic_rag

    return importlib.reload(agentic_rag)


def _run_async(coro):
    return asyncio.run(coro)


def stream_agent(agent, lc_messages: list) -> str:
    collected = ""
    token_box = st.empty()

    def handle_event(event: dict[str, Any]) -> None:
        nonlocal collected
        kind = event.get("event")
        if kind == "on_tool_start":
            collected = ""
            token_box.empty()
            return
        if kind not in {"on_chat_model_stream", "on_chat_model_stream_delta"}:
            return
        data = event.get("data") or {}
        chunk = data.get("chunk")
        if chunk is not None and getattr(chunk, "tool_call_chunks", None):
            return
        if chunk is not None and getattr(chunk, "tool_calls", None):
            return
        text = serialize_content(getattr(chunk, "content", "") if chunk else "")
        if text:
            collected += text
            token_box.markdown(collected)

    async def consume() -> None:
        astream_events = getattr(agent, "astream_events", None)
        if astream_events is None:
            return
        async for event in astream_events(
            {"messages": lc_messages},
            version="v2",
        ):
            handle_event(event)

    _run_async(consume())

    if not collected:
        result = agent.invoke({"messages": lc_messages})
        collected = extract_answer(result.get("messages", []))
        token_box.markdown(collected)

    return collected


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_pdfs" not in st.session_state:
        st.session_state.ingested_pdfs = []


def sync_uploaded_pdfs(uploaded_files, rag) -> None:
    current_ids = [f"{item.name}:{item.size}" for item in uploaded_files]
    if current_ids == st.session_state.ingested_pdfs:
        return

    rag.clear_documents()
    st.session_state.ingested_pdfs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

    for uploaded in uploaded_files:
        reader = PdfReader(BytesIO(uploaded.getvalue()))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"[{uploaded.name} p.{index}]\n{text}")
        if not pages:
            raise ValueError(f"Aucun texte extractible dans {uploaded.name}.")
        pieces = splitter.split_text("\n\n".join(pages))
        rag.vectorstore.add_texts(
            texts=pieces,
            metadatas=[{"source": uploaded.name} for _ in pieces],
        )
        st.session_state.ingested_pdfs.append(f"{uploaded.name}:{uploaded.size}")


def sidebar(rag) -> None:
    with st.sidebar:
        st.title("Agentic RAG")
        st.caption("Déposez un PDF, puis posez vos questions.")

        uploaded_files = st.file_uploader(
            "Déposer un PDF",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if rag is not None:
            if uploaded_files:
                try:
                    sync_uploaded_pdfs(uploaded_files, rag)
                    names = ", ".join(item.name for item in uploaded_files)
                    st.caption(f"Indexé : {names}")
                except Exception as exc:
                    st.error(str(exc))
            elif st.session_state.ingested_pdfs:
                rag.clear_documents()
                st.session_state.ingested_pdfs = []

        if st.button("Nouvelle conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown("Exemples")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True):
                st.session_state.pending_prompt = example
                st.rerun()


def main() -> None:
    init_state()
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    rag = None

    if not api_key:
        sidebar(None)
        st.header("Assistant RAG agentique")
        st.write("Déposez un PDF dans la barre latérale, puis posez une question.")
        st.warning("Ajoutez `OPENAI_API_KEY` dans un fichier `.env` à la racine du projet.")
        return

    try:
        rag = load_rag(api_key)
    except Exception as exc:
        sidebar(None)
        st.header("Assistant RAG agentique")
        st.error(f"Impossible de charger l’agent : {exc}")
        return

    sidebar(rag)
    st.header("Assistant RAG agentique")
    st.write("Les réponses viennent uniquement des PDF déposés.")
    agent = rag.agent
    chat_ready = bool(st.session_state.ingested_pdfs)

    if not chat_ready:
        st.info("Déposez au moins un PDF dans la barre latérale pour poser une question.")

    pending = st.session_state.pop("pending_prompt", None)
    if chat_ready and pending:
        st.session_state.messages.append({"role": "user", "content": pending})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if (
        chat_ready
        and st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
    ):
        with st.chat_message("assistant"):
            try:
                answer = stream_agent(
                    agent, to_langchain_messages(st.session_state.messages)
                )
            except Exception as exc:
                answer = f"Erreur : {exc}"
                st.error(str(exc))
            if not answer:
                answer = "L’agent n’a renvoyé aucune réponse."
                st.warning(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

    prompt = st.chat_input("Votre question…", disabled=not chat_ready)
    if chat_ready and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()


if __name__ == "__main__":
    main()
