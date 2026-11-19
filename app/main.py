import asyncio
import json
import threading
import traceback
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.backend import (
    CACHE_TTL_SECONDS,
    resume_travel_agent,
    run_travel_agent,
    stream_travel_agent,
)


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# =========================================================
# FASTAPI APPLICATION
# =========================================================

class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


app = FastAPI(
    title="VoyageCraft AI — Modular Travel Agent",
    description="Multi-Agent AI Travel Planner with Streaming and Human Approval",
    version="1.0.0",
)

app.mount("/static", NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =========================================================
# REQUEST MODELS
# =========================================================

class TravelRequest(BaseModel):
    query: str = Field(min_length=1, max_length=7500)
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = Field(default="", max_length=1000)


# =========================================================
# SSE SERIALIZATION
# =========================================================

def sse_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _enqueue_from_thread(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any] | None],
    item: dict[str, Any] | None,
) -> None:
    if loop.is_closed():
        return
    try:
        loop.call_soon_threadsafe(queue.put_nowait, item)
    except RuntimeError:
        pass


# =========================================================
# FRONTEND ROUTE
# =========================================================

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "app_name": "VoyageCraft AI",
        },
    )


# =========================================================
# STREAMING API
# =========================================================

@app.post("/api/run")
async def run(request_data: TravelRequest, request: Request):
    query = request_data.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    stop_event = threading.Event()

    def worker() -> None:
        try:
            for event in stream_travel_agent(
                user_input=query,
                thread_id=request_data.thread_id,
            ):
                if stop_event.is_set():
                    break
                _enqueue_from_thread(loop, queue, event)
        except Exception as exc:
            traceback.print_exc()
            if not stop_event.is_set():
                _enqueue_from_thread(
                    loop,
                    queue,
                    {
                        "type": "error",
                        "message": "Travel planning failed.",
                        "error": str(exc),
                    },
                )
        finally:
            _enqueue_from_thread(loop, queue, None)

    async def event_generator():
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            while True:
                if await request.is_disconnected():
                    stop_event.set()
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                if event is None:
                    break

                yield sse_event(event)
        finally:
            stop_event.set()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# =========================================================
# NON-STREAMING API
# =========================================================

@app.post("/api/travel")
def travel(request_data: TravelRequest):
    try:
        result = run_travel_agent(
            user_input=request_data.query.strip(),
            thread_id=request_data.thread_id,
        )
        return {"success": True, **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# =========================================================
# HUMAN APPROVAL API
# =========================================================

@app.post("/api/approval")
@app.post("/api/travel/approve")
def approval(request_data: ApprovalRequest):
    try:
        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )
        return {"success": True, **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "application": "Modular Travel Agent",
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "features": [
            "multi_agent_travel_planning",
            "langgraph_streaming",
            "custom_agent_events",
            "server_sent_events",
            "human_approval",
            "checkpoint_persistence",
            "agent_telemetry",
            "query_cache",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8020,
        reload=True,
    )