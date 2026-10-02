from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Agentic RAG", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_agent = None
_agent_error: str | None = None


def get_agent():
    global _agent, _agent_error
    if _agent is not None:
        return _agent
    if _agent_error:
        raise RuntimeError(_agent_error)
    try:
        from agentic_rag import agent

        _agent = agent
        return _agent
    except Exception as exc:
        _agent_error = str(exc)
        raise


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


def to_langchain_messages(messages: list[ChatMessage]):
    converted = []
    for message in messages:
        if message.role == "user":
            converted.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            converted.append(AIMessage(content=message.content))
    return converted


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


def extract_trace(messages: list) -> dict[str, Any]:
    tools: list[dict[str, str]] = []
    answer = ""
    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                tools.append(
                    {
                        "name": call.get("name", "outil"),
                        "args": json.dumps(call.get("args", {}), ensure_ascii=False),
                    }
                )
        if isinstance(message, ToolMessage):
            tools.append(
                {
                    "name": getattr(message, "name", None) or "résultat",
                    "args": serialize_content(message.content)[:400],
                }
            )
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            answer = serialize_content(message.content)
    return {"answer": answer, "tools": tools}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    try:
        get_agent()
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/chat")
def chat(payload: ChatRequest):
    try:
        agent = get_agent()
        result = agent.invoke({"messages": to_langchain_messages(payload.messages)})
        trace = extract_trace(result.get("messages", []))
        if not trace["answer"]:
            raise HTTPException(status_code=502, detail="Réponse vide de l'agent.")
        return trace
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest):
    try:
        agent = get_agent()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def generate():
        collected = ""
        try:
            async for event in agent.astream_events(
                {"messages": to_langchain_messages(payload.messages)},
                version="v2",
            ):
                kind = event.get("event")
                data = event.get("data") or {}
                if kind == "on_tool_start":
                    yield sse(
                        "tool",
                        {
                            "name": event.get("name") or "outil",
                            "status": "start",
                            "args": json.dumps(data.get("input", {}), ensure_ascii=False),
                        },
                    )
                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    text = serialize_content(getattr(chunk, "content", "") if chunk else "")
                    if text:
                        collected += text
                        yield sse("token", {"text": text})
            if not collected:
                result = agent.invoke({"messages": to_langchain_messages(payload.messages)})
                trace = extract_trace(result.get("messages", []))
                yield sse("done", trace)
            else:
                yield sse("done", {"answer": collected, "tools": []})
        except Exception as exc:
            yield sse("error", {"detail": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")
