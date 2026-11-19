from pathlib import Path
import json
import asyncio
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from backend import run_travel_agent, resume_travel_agent

# Allows existing synchronous agent functions to run alongside async FastApi/MCP calls
import nest_asyncio

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="VoyageCraft AI",
    description=(
        "LangGraph Multi-Agent Travel Planner with Supervisor, Guardrails, "
        "Human-in-the-Loop, Stream Events, and FastAPI Frontend"
    ),
    version="2.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        print("ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


@app.get("/api/travel/stream")
async def stream_travel_planner(message: str, thread_id: str | None = None):
    """
    Streams events (Supervisor reasoning, sub-agent progress, guardrails check, and status)
    using Server-Sent Events (SSE).
    """
    async def event_generator():
        try:
            user_message = message.strip()
            if not user_message:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Message cannot be empty.'})}\n\n"
                return

            # Example: Initializing event stream
            yield f"data: {json.dumps({'type': 'status', 'content': 'Checking Guardrails...'})}\n\n"
            await asyncio.sleep(0.1)

            # NOTE: If your backend `run_travel_agent` supports LangGraph stream mode (e.g., graph.astream_events or graph.stream),
            # you can yield chunks directly here:
            #
            # async for event in run_travel_agent_stream(user_message, thread_id):
            #     yield f"data: {json.dumps(event)}\n\n"

            # Executing full agent pipeline
            result = await asyncio.to_thread(
                run_travel_agent,
                user_input=user_message,
                thread_id=thread_id,
            )

            # Stream result complete event
            yield f"data: {json.dumps({'type': 'complete', 'data': result})}\n\n"

        except Exception as exc:
            print("STREAM ERROR:", exc)
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/travel/approve")
async def approve_travel_plan(request_data: ApprovalRequest):
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                },
            )

        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        print("APPROVAL ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "VoyageCraft AI API is running",
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "human_in_the_loop",
            "stream_events",
        ],
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8020,
        reload=True,
        loop="asyncio",
    )