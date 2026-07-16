"""FastAPI backend.

    POST /api/upload   ingest a document ({title, text} or an uploaded file)
    POST /api/chat     ask a question -> grounded answer + citations + agents used
    GET  /api/stats    engine stats
    GET  /api/health   liveness
    GET  /             built-in chat UI (React via CDN, no build step)

Mirrors the decoupled architecture (Next.js frontend ⇄ FastAPI backend). An
Analyst is pre-seeded with the sample documents at startup so the chat works
immediately. FastAPI/uvicorn are optional; import this module only when serving.
"""

from __future__ import annotations

from typing import Optional

from ..config import Settings
from ..engine import Analyst
from ..mockdata import sample_documents


def create_app(seed: bool = True, analyst: Optional[Analyst] = None):
    from fastapi import FastAPI, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="Veritas", version="0.1.0",
                  description="Domain-specific AI analyst — multi-agent RAG over your documents")

    if analyst is None:
        analyst = Analyst(Settings())
        if seed:
            for doc in sample_documents():
                analyst.ingest(doc)
    app.state.analyst = analyst

    class UploadBody(BaseModel):
        title: str = "Untitled"
        text: str

    class ChatBody(BaseModel):
        query: str

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/api/upload")
    def upload(body: UploadBody) -> dict:
        rep = app.state.analyst.ingest_text(body.text, title=body.title)
        return {"doc_id": rep.doc_id, "title": rep.title, "chunks": rep.chunks, "pages": rep.pages}

    @app.post("/api/upload-file")
    async def upload_file(file: UploadFile) -> dict:
        raw = (await file.read()).decode("utf-8", errors="replace")
        rep = app.state.analyst.ingest_text(raw, title=file.filename or "upload")
        return {"doc_id": rep.doc_id, "title": rep.title, "chunks": rep.chunks, "pages": rep.pages}

    @app.post("/api/chat")
    def chat(body: ChatBody) -> JSONResponse:
        ans = app.state.analyst.ask(body.query)
        return JSONResponse(ans.to_dict())

    @app.get("/api/stats")
    def stats() -> dict:
        return app.state.analyst.stats()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        from .ui import CHAT_HTML
        return CHAT_HTML

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
