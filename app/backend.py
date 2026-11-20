import threading
import time
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.config import (
    MAX_USER_INPUT_CHARS,
    TOKEN_LIMIT,
)
from app.graph.checkpoint import create_checkpointer
from app.graph.workflow import build_workflow


# Initialize once when the module loads.
db_pool, checkpointer = create_checkpointer()

travel_graph = build_workflow(checkpointer)


# =========================================================
# QUERY-LEVEL TTL CACHE
# =========================================================

_QUERY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()
CACHE_TTL_SECONDS = 3600  # 1 hour


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def get_cached_result(query: str) -> dict[str, Any] | None:
    key = _normalize_query(query)
    with _CACHE_LOCK:
        entry = _QUERY_CACHE.get(key)
        if not entry:
            return None

        created_at, payload = entry
        if time.time() - created_at > CACHE_TTL_SECONDS:
            _QUERY_CACHE.pop(key, None)
            return None

        return dict(payload)


def set_cached_result(query: str, serialized: dict[str, Any]) -> None:
    # Never cache blocked requests, empty answers, or runs paused for human approval.
    if (
        serialized.get("guardrail_allowed") is False
        or serialized.get("requires_approval")
        or not str(serialized.get("answer", "")).strip()
    ):
        return

    key = _normalize_query(query)
    with _CACHE_LOCK:
        _QUERY_CACHE[key] = (time.time(), dict(serialized))


def create_initial_state(user_input):

    return {
        "messages": [
            HumanMessage(content=user_input)
        ],

        "user_query": user_input,

        "guardrail_allowed": True,
        "guardrail_reason": "",

        "selected_agents": [],
        "trip_constraints": {},

        "supervisor_reasoning": "",

        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",

        "itinerary": "",
        "prev_itinerary": "",
        "revision_count": 0,
        
        "approval_request": "",

        "approved": False,
        "human_feedback": "",

        "final_response": "",

        "llm_calls": 0,
        "token_usage": 0,
        "token_limit": TOKEN_LIMIT,

        "agent_metrics": {},
        "overall_latency_ms": 0.0,
        "total_cost_usd": 0.0,
        "total_cost_inr": 0.0,
        "overall_accuracy": 100.0,
    }


def _extract_node_progress(payload: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return events

    for node_name, node_state in payload.items():
        if node_name == "__interrupt__" or not isinstance(node_state, dict):
            continue

        clean_agent = str(node_name).removesuffix("_agent")
        detail = ""
        queued_agents = node_state.get("selected_agents") or []

        if node_state.get("supervisor_reasoning"):
            detail = str(node_state["supervisor_reasoning"])
        elif node_state.get("flight_results"):
            detail = "Flight options gathered."
        elif node_state.get("hotel_results"):
            detail = "Boutique & stay options gathered."
        elif node_state.get("weather_results"):
            detail = "Destination weather checked."
        elif node_state.get("budget_results"):
            detail = "Budget breakdown calculated."
        elif node_state.get("itinerary"):
            detail = "Draft itinerary generated."
        elif "guardrail_allowed" in node_state:
            detail = (
                "Request passed guardrails."
                if node_state["guardrail_allowed"]
                else (node_state.get("guardrail_reason") or "Blocked by guardrails.")
            )

        events.append({
            "type": "stage",
            "id": clean_agent,
            "agent": node_name,
            "label": clean_agent.replace("_", " ").title(),
            "status": "completed",
            "detail": detail or f"{clean_agent.replace('_', ' ').title()} finished.",
            "selected_agents": queued_agents,
            "supervisor_reasoning": node_state.get("supervisor_reasoning"),
            "guardrail_allowed": node_state.get("guardrail_allowed"),
        })

        # Mark supervisor-selected agents as "running" right after routing completes
        if clean_agent == "supervisor" and queued_agents:
            for agent_name in queued_agents:
                sub_id = str(agent_name).removesuffix("_agent")
                events.append({
                    "type": "stage",
                    "id": sub_id,
                    "agent": agent_name,
                    "label": sub_id.replace("_", " ").title(),
                    "status": "running",
                    "detail": "Gathering live travel data...",
                })

    return events


def validate_input(user_input):

    if not isinstance(user_input, str):
        raise ValueError("Input must be a string.")

    user_input = user_input.strip()

    if not user_input:
        raise ValueError("Input cannot be empty.")

    if len(user_input) > MAX_USER_INPUT_CHARS:
        raise ValueError("Input exceeds maximum length.")

    return user_input


def _compute_fallback_accuracy(data: dict[str, Any]) -> float:
    if data.get("guardrail_allowed") is False:
        return 0.0
    selected = data.get("selected_agents") or []
    result_map = {
        "flight_agent": data.get("flight_results"),
        "hotel_agent": data.get("hotel_results"),
        "weather_agent": data.get("weather_results"),
        "budget_agent": data.get("budget_results"),
        "itinerary_agent": data.get("itinerary") or data.get("final_response"),
    }
    if not selected:
        return 100.0
    completed = sum(1 for agent in selected if str(result_map.get(agent, "")).strip())
    return round((completed / len(selected)) * 100, 1)


def _extract_interrupts(snapshot: Any, data: dict[str, Any], streamed_interrupts: list[Any] | None = None) -> list[Any]:
    # 1. Parent snapshot interrupts (LangGraph StateSnapshot.interrupts)
    snap_interrupts = list(getattr(snapshot, "interrupts", None) or [])
    if snap_interrupts:
        return snap_interrupts

    # 2. Subgraph task interrupts if get_state(..., subgraphs=True) was used
    for task in getattr(snapshot, "tasks", None) or []:
        task_interrupts = list(getattr(task, "interrupts", None) or [])
        if task_interrupts:
            return task_interrupts

    # 3. Interrupts captured directly from stream updates
    if streamed_interrupts:
        return list(streamed_interrupts)

    # 4. Legacy __interrupt__ key in result dict
    return list(data.get("__interrupt__", []) or [])


def serialize_result(
    result,
    thread_id,
    latency_ms: float = 0.0,
    snapshot: Any = None,
    streamed_interrupts: list[Any] | None = None,
):

    data = result or {}
    interrupt_list = _extract_interrupts(snapshot, data, streamed_interrupts)

    interrupt_payload = None
    if interrupt_list:
        first_interrupt = interrupt_list[0]
        interrupt_payload = getattr(first_interrupt, "value", first_interrupt)

    requires_approval = bool(interrupt_payload)

    if isinstance(interrupt_payload, dict):
        answer = (
            interrupt_payload.get("draft_itinerary")
            or interrupt_payload.get("itinerary")
            or data.get("itinerary")
            or data.get("final_response", "")
        )
        approval_req = (
            interrupt_payload.get("message")
            or interrupt_payload.get("approval_request")
            or data.get("approval_request", "")
        )
    else:
        answer = data.get("final_response") or data.get("itinerary", "")
        approval_req = (
            str(interrupt_payload)
            if interrupt_payload
            else data.get("approval_request", "")
        )

    token_usage = int(data.get("token_usage", 0) or 0)
    agent_metrics = data.get("agent_metrics") or {}

    raw_latency_ms = float(data.get("overall_latency_ms") or latency_ms or 0.0)
    if latency_ms and raw_latency_ms == 0.0:
        raw_latency_ms = float(latency_ms)

    total_cost_usd = round(float(data.get("total_cost_usd", 0.0) or 0.0), 6)
    total_cost_inr = round(float(data.get("total_cost_inr", total_cost_usd * 84.0) or 0.0), 4)

    if "overall_accuracy" in data and data["overall_accuracy"] is not None:
        overall_accuracy = round(float(data["overall_accuracy"]), 1)
    else:
        overall_accuracy = _compute_fallback_accuracy(data)

    return {
        "thread_id": thread_id,
        "answer": answer,
        "cached": False,

        "requires_approval": requires_approval,
        "approval_request": approval_req,

        "flight_results": data.get("flight_results", ""),
        "hotel_results": data.get("hotel_results", ""),
        "weather_results": data.get("weather_results", ""),
        "budget_results": data.get("budget_results", ""),


        "itinerary": data.get("itinerary", ""),
        "prev_itinerary": data.get("prev_itinerary", ""),
        "revision_count": data.get("revision_count", 0),

        "selected_agents": data.get("selected_agents", []),
        "trip_constraints": data.get("trip_constraints", {}),
        "supervisor_reasoning": data.get("supervisor_reasoning", ""),

        "guardrail_allowed": data.get("guardrail_allowed", True),
        "guardrail_reason": data.get("guardrail_reason", ""),

        "approved": data.get("approved"),
        "human_feedback": data.get("human_feedback", ""),

        "llm_calls": int(data.get("llm_calls", 0) or 0),
        "token_usage": token_usage,
        "token_limit": TOKEN_LIMIT,
        "remaining_tokens": max(0, TOKEN_LIMIT - token_usage),

        "agent_metrics": agent_metrics,
        "overall_latency_ms": round(raw_latency_ms, 1),
        "overall_latency_seconds": round(raw_latency_ms / 1000, 2),
        "total_cost_usd": total_cost_usd,
        "total_cost_inr": total_cost_inr,
        "overall_accuracy": overall_accuracy,
    }


def run_travel_agent(user_input, thread_id=None):

    user_input = validate_input(user_input)
    thread_id = f"user_{uuid.uuid4().hex}"

    started_at = time.perf_counter()
    cached = get_cached_result(user_input)
    if cached is not None:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        cached.update({
            "thread_id": thread_id,
            "cached": True,
            "overall_latency_ms": elapsed_ms,
            "overall_latency_seconds": round(elapsed_ms / 1000, 2),
        })
        return cached

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        create_initial_state(user_input),
        config=config,
    )

    snapshot = travel_graph.get_state(config, subgraphs=True)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    serialized = serialize_result(
        result,
        thread_id,
        latency_ms=elapsed_ms,
        snapshot=snapshot,
    )
    set_cached_result(user_input, serialized)

    return serialized


def resume_travel_agent(
    thread_id,
    approved,
    feedback="",
):

    if not thread_id:
        raise ValueError("thread_id is required.")

    if not isinstance(approved, bool):
        raise ValueError("approved must be boolean.")

    if not isinstance(feedback, str):
        raise ValueError("Feedback must be a string.")

    if len(feedback) > 1000:
        raise ValueError("Feedback is too long.")

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    started_at = time.perf_counter()
    print(approved , '\n \n', feedback)
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback,
            }
        ),
        config=config,
    )

    snapshot = travel_graph.get_state(config, subgraphs=True)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)

    serialized = serialize_result(
        result,
        thread_id,
        latency_ms=elapsed_ms,
        snapshot=snapshot,
    )

    # Cache the final approved itinerary under its original user query
    original_query = (result or {}).get("user_query", "")
    if original_query:
        set_cached_result(original_query, serialized)

    return serialized


def stream_travel_agent(user_input: str, thread_id: str | None = None):
    user_input = validate_input(user_input)
    # Fresh thread_id per new request prevents stale checkpoints from colliding.
    thread_id = f"user_{uuid.uuid4().hex}"

    started_at = time.perf_counter()

    yield {
        "type": "run_started",
        "thread_id": thread_id,
        "topic": user_input,
    }

    cached = get_cached_result(user_input)
    if cached is not None:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        cached.update({
            "thread_id": thread_id,
            "cached": True,
            "overall_latency_ms": elapsed_ms,
            "overall_latency_seconds": round(elapsed_ms / 1000, 2),
        })
        print(f"[CACHE HIT] query={_normalize_query(user_input)!r}")

        yield {
            "type": "status",
            "thread_id": thread_id,
            "content": "Loaded itinerary from cache.",
        }

        yield {
            "type": "final",
            "thread_id": thread_id,
            "result": cached,
        }

        yield {
            "type": "done",
            "thread_id": thread_id,
        }
        return

    print(f"[CACHE MISS] query={_normalize_query(user_input)!r}")

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    streamed_interrupts: list[Any] = []

    for part in travel_graph.stream(
        create_initial_state(user_input),
        config=config,
        stream_mode=["updates", "custom"],
        subgraphs=True,
        version="v2",
    ):
        part_type = part.get("type") if isinstance(part, dict) else None
        payload = part.get("data") if isinstance(part, dict) else None

        if part_type == "custom" and isinstance(payload, dict):
            event_name = payload.get("event", "progress")
            event_data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
            agent_name = event_data.get("agent") or payload.get("agent")

            if agent_name:
                yield {
                    "type": "agent_event",
                    "thread_id": thread_id,
                    "id": str(agent_name).removesuffix("_agent"),
                    "agent": agent_name,
                    "event": event_name,
                    "status": "completed" if "completed" in str(event_name) else "running",
                    "message": payload.get("message", "Working..."),
                    "data": event_data,
                }
            else:
                yield {
                    "type": "status",
                    "thread_id": thread_id,
                    "content": payload.get("message", "Working..."),
                }

        elif part_type == "updates":
            if isinstance(payload, dict) and "__interrupt__" in payload:
                raw_intr = payload["__interrupt__"]
                if isinstance(raw_intr, (list, tuple)):
                    streamed_interrupts.extend(raw_intr)
                elif raw_intr:
                    streamed_interrupts.append(raw_intr)

            for stage_evt in _extract_node_progress(payload):
                stage_evt["thread_id"] = thread_id
                yield stage_evt

    snapshot = travel_graph.get_state(config, subgraphs=True)
    result = snapshot.values if snapshot else {}
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)

    serialized = serialize_result(
        result,
        thread_id,
        latency_ms=elapsed_ms,
        snapshot=snapshot,
        streamed_interrupts=streamed_interrupts,
    )
    set_cached_result(user_input, serialized)

    yield {
        "type": "final",
        "thread_id": thread_id,
        "result": serialized,
    }

    yield {
        "type": "done",
        "thread_id": thread_id,
    }