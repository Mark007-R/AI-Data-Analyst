"""DataAI — FastAPI application.

Run locally:  uvicorn app.main:app --reload --port 7860
"""
from __future__ import annotations

import re
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .ingest import IngestError, ingest
from .mcp_client import manager
from .pipeline import STATUS, run_pipeline, set_status

DATASET_ID_RE = re.compile(r"^[a-z0-9]{8,32}\Z")   # \Z so a trailing newline can't slip past
CHART_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}\Z")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await manager.start()
    except Exception as e:
        print(f"[mcp] failed to start MCP server (chat will be degraded): {e}")
    yield
    await manager.stop()


app = FastAPI(title="DataAI", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


def _check_dataset_id(dataset_id: str) -> None:
    if not DATASET_ID_RE.match(dataset_id):
        raise HTTPException(400, "Invalid dataset id")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    # file.size is the real byte count Starlette accumulated while parsing — check it
    # before reading the whole body into RAM.
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (limit: 50 MB).")
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (limit: 50 MB).")
    try:
        meta = await anyio.to_thread.run_sync(ingest, raw, file.filename or "upload.csv")
    except IngestError as e:
        raise HTTPException(400, str(e))
    dataset_id = meta["dataset_id"]
    set_status(dataset_id, "Queued")

    async def _run():
        await anyio.to_thread.run_sync(run_pipeline, dataset_id)

    import asyncio
    asyncio.get_running_loop().create_task(_run())
    return {"dataset_id": dataset_id, "tables": meta["tables"]}


@app.get("/api/status/{dataset_id}")
async def status(dataset_id: str):
    _check_dataset_id(dataset_id)
    st = STATUS.get(dataset_id)
    if st is None:
        # server restarted: report exists -> done, else unknown
        if config.dataset_report_path(dataset_id).exists():
            return {"stage": "Done", "done": True, "error": None}
        raise HTTPException(404, "Unknown dataset")
    return st


@app.get("/api/report/{dataset_id}")
async def report(dataset_id: str):
    _check_dataset_id(dataset_id)
    p = config.dataset_report_path(dataset_id)
    if not p.exists():
        raise HTTPException(404, "Report not ready")
    return FileResponse(p, media_type="application/json")  # streamed off-loop, no re-parse


@app.post("/api/chat/{dataset_id}")
async def chat(dataset_id: str, req: ChatRequest):
    _check_dataset_id(dataset_id)
    if not config.dataset_meta_path(dataset_id).exists():
        raise HTTPException(404, "Unknown dataset")
    if not req.message.strip():
        raise HTTPException(400, "Empty message")
    from .chat import answer
    try:
        return await answer(dataset_id, req.message.strip(), req.history)
    except Exception:
        import traceback
        traceback.print_exc()  # detail to logs; generic text to the client
        return {"answer": "Sorry — the analyst hit an unexpected internal error. "
                          "Please try again.", "charts": []}


@app.get("/api/chart/{dataset_id}/{chart_id}")
async def chart(dataset_id: str, chart_id: str):
    _check_dataset_id(dataset_id)
    if not CHART_ID_RE.match(chart_id):
        raise HTTPException(400, "Invalid chart id")
    p = config.dataset_charts_dir(dataset_id) / f"{chart_id}.json"
    if not p.exists():
        raise HTTPException(404, "Chart not found")
    return FileResponse(p, media_type="application/json")


@app.get("/api/health")
async def health():
    return {"ok": True, "llm_enabled": config.LLM_ENABLED, "mcp_ready": manager.ready}


@app.get("/")
async def index():
    return FileResponse(config.STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
