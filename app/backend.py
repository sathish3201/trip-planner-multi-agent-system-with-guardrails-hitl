
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from typing import Any


from app.config import (
    TOKEN_LIMIT,
    MAX_USER_INPUT_CHARS,
)

from app.graph.checkpoint import create_checkpointer
from app.graph.workflow import build_workflow


# Initialize once when the module loads.
db_pool, checkpointer = create_checkpointer()

travel_graph = build_workflow(checkpointer)


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
        "approval_request": "",

        "approved": False,
        "human_feedback": "",

        "final_response": "",

        "llm_calls": 0,
        "token_usage": 0,
        "token_limit": TOKEN_LIMIT,
    }

def _extract_node_progress(payload: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return events

    for node_name, node_state in payload.items():
        if not isinstance(node_state, dict):
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


def serialize_result(result, thread_id):

    interrupt_list = result.get("__interrupt__", [])

    interrupt_payload = None

    if interrupt_list:

        interrupt_payload = getattr(
            interrupt_list[0],
            "value",
            {},
        )

    answer = (
        interrupt_payload.get("draft_itinerary", "")
        if interrupt_payload
        else result.get("final_response", "")
    )

    token_usage = result.get("token_usage", 0)

    return {
        "thread_id": thread_id,
        "answer": answer,

        "requires_approval": bool(interrupt_payload),

        "approval_request": result.get(
            "approval_request", ""
        ),

        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),

        "itinerary": result.get("itinerary", ""),

        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),

        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),

        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),

        "llm_calls": result.get("llm_calls", 0),
        "token_usage": token_usage,
        "token_limit": TOKEN_LIMIT,
        "remaining_tokens": max(0, TOKEN_LIMIT - token_usage),
    }


def run_travel_agent(user_input, thread_id=None):

    user_input = validate_input(user_input)

    thread_id = thread_id or f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        create_initial_state(user_input),
        config=config,
    )

    return serialize_result(result, thread_id)


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

    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback,
            }
        ),
        config=config,
    )

    return serialize_result(result, thread_id)


# def stream_travel_agent(user_input, thread_id=None):

#     user_input = validate_input(user_input)

#     thread_id = thread_id or f"user_{uuid.uuid4().hex}"

#     config = {
#         "configurable": {
#             "thread_id": thread_id
#         }
#     }

#     yield {
#         "type": "run_started",
#         "thread_id": thread_id,
#         "topic": user_input,
#     }

#     for part in travel_graph.stream(
#         create_initial_state(user_input),
#         config=config,
#         stream_mode=["updates", "custom"],
#         subgraphs=True,
#         version="v2",
#     ):

#         if part["type"] != "custom":
#             continue

#         event = part["data"]

#         yield {
#             "type": "agent_event",
#             "thread_id": thread_id,
#             "event": event.get("event"),
#             "message": event.get("message"),
#             "data": event.get("data", {}),
#         }

#     snapshot = travel_graph.get_state(config)

#     result = snapshot.values

#     yield {
#         "type": "final",
#         "thread_id": thread_id,
#         "result": serialize_result(result, thread_id),
#     }

#     yield {
#         "type": "done",
#         "thread_id": thread_id,
#     }


def stream_travel_agent(user_input: str, thread_id: str | None = None):
    user_input = validate_input(user_input)
    thread_id = thread_id or f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    yield {
        "type": "run_started",
        "thread_id": thread_id,
        "topic": user_input,
    }

    yield {
        "type": "stage",
        "id": "guardrail",
        "agent": "guardrail",
        "label": "Guardrails",
        "status": "running",
        "detail": "Checking request safety and constraints...",
    }

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
            agent_key = payload.get("agent") or payload.get("event") or "agent"
            yield {
                "type": "agent_event",
                "thread_id": thread_id,
                "id": str(agent_key).removesuffix("_agent"),
                "agent": agent_key,
                "event": payload.get("event", "progress"),
                "status": payload.get("status", "running"),
                "message": payload.get("message", "Working..."),
                "data": payload.get("data", {}),
            }

        elif part_type == "updates":
            for stage_evt in _extract_node_progress(payload):
                stage_evt["thread_id"] = thread_id
                yield stage_evt

    snapshot = travel_graph.get_state(config)
    result = snapshot.values if snapshot else {}

    yield {
        "type": "final",
        "thread_id": thread_id,
        "result": serialize_result(result, thread_id),
    }

    yield {
        "type": "done",
        "thread_id": thread_id,
    }