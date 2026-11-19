
from langgraph.config import get_stream_writer

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from pydantic import BaseModel, Field
from typing import Literal

from app.utils.token_utils import budgeted_invoke
from app.utils.context import (
    limit_context,
    compact_records,
    build_agent_context,
)

from app.prompts import (
    GUARDRAIL_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    FLIGHT_SYSTEM_PROMPT,
    HOTEL_SYSTEM_PROMPT,
    WEATHER_SYSTEM_PROMPT,
    BUDGET_SYSTEM_PROMPT,
    ITINERARY_SYSTEM_PROMPT,
    FINAL_SYSTEM_PROMPT,
    supervisor_prompt,
    flight_prompt,
    hotel_prompt,
    weather_prompt,
    budget_prompt,
    itinerary_prompt,
    final_prompt,
)

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)

from app.utils.async_utils import run_async


# =====================================================
# STREAMING EVENT HELPER
# =====================================================

def emit_event(event_type, message, data=None):

    try:
        writer = get_stream_writer()

        writer({
            "event": event_type,
            "message": message,
            "data": data or {},
        })

    except Exception:
        pass


# =====================================================
# PYDANTIC SCHEMAS
# =====================================================

class GuardrailResult(BaseModel):
    allowed: bool
    reason: str = ""


class TripConstraints(BaseModel):
    destination: str = ""
    origin: str = ""
    duration: str = ""
    budget: str = ""
    travel_style: str = ""
    special_preferences: list[str] = Field(
        default_factory=list
    )


class SupervisorResult(BaseModel):

    selected_agents: list[
        Literal[
            "flight_agent",
            "hotel_agent",
            "weather_agent",
            "budget_agent",
            "itinerary_agent",
        ]
    ] = Field(
        default_factory=lambda: ["itinerary_agent"]
    )

    trip_constraints: TripConstraints = Field(
        default_factory=TripConstraints
    )

    reasoning: str = ""


# =====================================================
# STRUCTURED OUTPUT
# =====================================================

def structured_invoke(
    agent_name,
    messages,
    current_usage,
    schema,
    max_output_tokens=500,
):

    structured_messages = list(messages)

    structured_messages.append(
        SystemMessage(
            content=(
                "Return exactly one valid JSON object. "
                "Follow this schema:\n"
                f"{schema.model_json_schema()}"
            )
        )
    )

    updated_usage = current_usage

    for attempt in range(2):

        try:

            response, updated_usage = budgeted_invoke(
                agent_name,
                structured_messages,
                updated_usage,
                max_output_tokens=max_output_tokens,
                response_format={"type": "json_object"},
                temperature=0,
            )

            import json

            parsed = json.loads(response.content)

            return schema.model_validate(parsed), updated_usage

        except Exception:

            if attempt == 0:

                structured_messages.append(
                    HumanMessage(
                        content=(
                            "Return valid JSON only. "
                            "Correct the previous response."
                        )
                    )
                )

    raise ValueError(
        f"{agent_name}: Structured output validation failed."
    )


# =====================================================
# SUPERVISOR AGENT
# =====================================================

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def supervisor_agent(state):

    query = limit_context(
        state.get("user_query"),
        800,
    )

    token_usage = state.get("token_usage", 0)
    llm_calls = state.get("llm_calls", 0)

    emit_event(
        "agent_started",
        "Travel supervisor started."
    )

    try:

        guardrail, token_usage = structured_invoke(
            "guardrail",
            [
                SystemMessage(
                    content=GUARDRAIL_SYSTEM_PROMPT
                ),
                HumanMessage(content=query),
            ],
            token_usage,
            GuardrailResult,
            200,
        )

        llm_calls += 1

        if not guardrail.allowed:

            emit_event(
                "guardrail_blocked",
                "Travel request did not pass validation."
            )

            return {
                "guardrail_allowed": False,
                "guardrail_reason": guardrail.reason,
                "final_response": guardrail.reason,
                "selected_agents": [],
                "token_usage": token_usage,
                "llm_calls": llm_calls,
            }

        emit_event(
            "guardrail_passed",
            "Travel request passed validation."
        )

        result, token_usage = structured_invoke(
            "supervisor",
            [
                SystemMessage(
                    content=SUPERVISOR_SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=supervisor_prompt(query)
                ),
            ],
            token_usage,
            SupervisorResult,
            500,
        )

        llm_calls += 1

        selected = [
            name
            for name in AGENT_ORDER
            if name in result.selected_agents
        ]

        if "itinerary_agent" not in selected:
            selected.append("itinerary_agent")

        constraints = result.trip_constraints.model_dump()

        emit_event(
            "agents_selected",
            "Supervisor selected travel agents.",
            {
                "selected_agents": selected,
                "constraints": constraints,
            }
        )

        return {
            "guardrail_allowed": True,
            "guardrail_reason": guardrail.reason,
            "selected_agents": selected,
            "trip_constraints": constraints,
            "supervisor_reasoning": result.reasoning,
            "token_usage": token_usage,
            "llm_calls": llm_calls,
        }

    except Exception as exc:

        emit_event(
            "agent_error",
            "Supervisor execution failed.",
            {"error": str(exc)}
        )

        return {
            "guardrail_allowed": False,
            "final_response": "Unable to plan the travel request.",
            "selected_agents": [],
            "token_usage": token_usage,
            "llm_calls": llm_calls,
        }


# =====================================================
# FLIGHT AGENT
# =====================================================

def flight_agent(state):

    emit_event("agent_started", "Flight agent started.")

    constraints = state.get("trip_constraints", {})

    token_usage = state.get("token_usage", 0)
    llm_calls = state.get("llm_calls", 0)

    try:

        airports = run_async(
            aviation_mcp_call("list_airports")
        )

        airlines = run_async(
            aviation_mcp_call("list_airlines")
        )

        airport_data = compact_records(
            airports,
            ("name", "city", "country", "iata", "icao"),
            max_items=5,
            max_chars=700,
        )

        airline_data = compact_records(
            airlines,
            ("name", "iata", "icao"),
            max_items=5,
            max_chars=400,
        )

        prompt = flight_prompt(
            state.get("user_query", ""),
            constraints.get("origin", ""),
            constraints.get("destination", ""),
            airport_data,
            airline_data,
        )

        response, token_usage = budgeted_invoke(
            "flight_agent",
            [
                SystemMessage(content=FLIGHT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            token_usage,
            500,
        )

        llm_calls += 1
        result = response.content

    except Exception as exc:

        result = f"Flight information unavailable: {exc}"

    emit_event("agent_completed", "Flight agent completed.")

    return {
        "flight_results": limit_context(result, 1200),
        "token_usage": token_usage,
        "llm_calls": llm_calls,
    }


# =====================================================
# HOTEL AGENT
# =====================================================

def hotel_agent(state):

    emit_event("agent_started", "Hotel agent started.")

    constraints = state.get("trip_constraints", {})
    destination = constraints.get("destination", "")

    try:

        query = hotel_prompt(
            state.get("user_query", ""),
            destination,
        )

        raw_results = run_async(
            tavily_mcp_search(
                f"Best hotels in {destination}. {query}"
            )
        )

        result = compact_records(
            raw_results,
            ("title", "name", "content", "snippet", "url"),
            max_items=5,
            max_chars=1400,
        )

    except Exception as exc:
        result = f"Hotel information unavailable: {exc}"

    emit_event("agent_completed", "Hotel agent completed.")

    return {
        "hotel_results": limit_context(result, 1400)
    }


# =====================================================
# WEATHER AGENT
# =====================================================

def weather_agent(state):

    emit_event("agent_started", "Weather agent started.")

    constraints = state.get("trip_constraints", {})

    city = (
        constraints.get("destination")
        or extract_destination(state.get("user_query", ""))
    )

    try:

        current = run_async(weather_mcp_search(city))
        forecast = run_async(forecast_mcp_search(city))

        current_text = compact_records(
            current,
            ("city", "temperature", "condition", "description"),
            max_items=3,
            max_chars=400,
        )

        forecast_text = compact_records(
            forecast,
            ("date", "temperature", "condition", "description"),
            max_items=5,
            max_chars=500,
        )

        response = weather_prompt(
            city,
            current_text,
            forecast_text,
        )

    except Exception as exc:
        response = f"Weather information unavailable: {exc}"

    emit_event("agent_completed", "Weather agent completed.")

    return {
        "weather_results": limit_context(response, 900)
    }


# =====================================================
# BUDGET AGENT
# =====================================================

def budget_agent(state):

    emit_event("agent_started", "Budget agent started.")

    ctx = build_agent_context(state)

    token_usage = state.get("token_usage", 0)
    llm_calls = state.get("llm_calls", 0)

    try:

        response, token_usage = budgeted_invoke(
            "budget_agent",
            [
                SystemMessage(content=BUDGET_SYSTEM_PROMPT),
                HumanMessage(content=budget_prompt(ctx)),
            ],
            token_usage,
            650,
        )

        llm_calls += 1
        result = response.content

    except Exception as exc:
        result = f"Budget analysis unavailable: {exc}"

    emit_event("agent_completed", "Budget agent completed.")

    return {
        "budget_results": limit_context(result, 1000),
        "token_usage": token_usage,
        "llm_calls": llm_calls,
    }


# =====================================================
# ITINERARY AGENT
# =====================================================

def itinerary_agent(state):

    emit_event("agent_started", "Itinerary agent started.")

    ctx = build_agent_context(state)

    token_usage = state.get("token_usage", 0)
    llm_calls = state.get("llm_calls", 0)

    try:

        response, token_usage = budgeted_invoke(
            "itinerary_agent",
            [
                SystemMessage(content=ITINERARY_SYSTEM_PROMPT),
                HumanMessage(content=itinerary_prompt(ctx)),
            ],
            token_usage,
            1200,
        )

        llm_calls += 1
        result = response.content

    except Exception as exc:
        result = f"Unable to generate itinerary: {exc}"

    emit_event("itinerary_generated", "Draft itinerary generated.")

    return {
        "itinerary": limit_context(result, 2500),
        "approval_request": "Review and approve the itinerary.",
        "token_usage": token_usage,
        "llm_calls": llm_calls,
    }


# =====================================================
# HUMAN APPROVAL AGENT
# =====================================================

from langgraph.types import interrupt


def human_approval_agent(state):

    review = interrupt({
        "question": "Do you approve this itinerary?",
        "draft_itinerary": state.get("itinerary", ""),
        "approval_request": state.get("approval_request", ""),
        "expected_response": {
            "approved": True,
            "feedback": "Optional revision feedback",
        },
    })

    if not isinstance(review, dict):
        review = {}

    approved = review.get("approved", False)

    if not isinstance(approved, bool):
        raise ValueError("Approval must be boolean.")

    feedback = str(
        review.get("feedback", "")
    ).strip()

    if len(feedback) > 1000:
        raise ValueError("Feedback is too long.")

    emit_event(
        "approval_received",
        "Human approval received.",
        {"approved": approved}
    )

    return {
        "approved": approved,
        "human_feedback": feedback,
    }


# =====================================================
# FINAL AGENT
# =====================================================

def final_agent(state):

    draft = state.get("itinerary", "")
    token_usage = state.get("token_usage", 0)
    llm_calls = state.get("llm_calls", 0)

    if state.get("approved", False):

        result = draft

    else:

        try:

            response, token_usage = budgeted_invoke(
                "final_agent",
                [
                    SystemMessage(content=FINAL_SYSTEM_PROMPT),
                    HumanMessage(
                        content=final_prompt(
                            state.get("user_query", ""),
                            draft,
                            state.get("human_feedback", ""),
                        )
                    ),
                ],
                token_usage,
                1000,
            )

            llm_calls += 1
            result = response.content

        except Exception:
            result = draft + "\n\nRevision could not be generated."

    emit_event(
        "agent_completed",
        "Final travel itinerary ready."
    )

    return {
        "final_response": result,
        "token_usage": token_usage,
        "llm_calls": llm_calls,
        "messages": [
            AIMessage(content=result)
        ],
    }


# =====================================================
# GUARDRAIL BLOCKED
# =====================================================

def guardrail_blocked_agent(state):

    return {
        "final_response": state.get(
            "final_response",
            state.get("guardrail_reason", "Request blocked."),
        )
    }
