import os
import re
import json
import uuid
import math
import asyncio
import operator
import certifi
import nest_asyncio

from typing import Any, TypedDict, Annotated
from collections.abc import Mapping, Sequence


from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()



nest_asyncio.apply()

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt

from langchain_groq import ChatGroq
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from mcp_client import (
    _require_env,
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


# =========================================================
# CONFIGURATION
# =========================================================

GROQ_API_KEY = _require_env(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY"),
)

TOKEN_LIMIT = 6000

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0.2,
)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is missing.")

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = (
            f"{database_url}{separator}sslmode=require"
        )

    return database_url


# =========================================================
# ASYNC HELPER
# =========================================================

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


# =========================================================
# TOKEN MANAGEMENT
# =========================================================

def estimate_tokens(text: str) -> int:
    """
    Conservative approximate token estimation.
    This is not the exact Groq tokenizer.
    """

    return max(1, math.ceil(len(text) / 3))


def messages_to_text(messages) -> str:
    parts = []

    for message in messages:
        parts.append(str(message.content))

    return "\n".join(parts)


def get_actual_token_usage(response, estimated_input: int,
                           reserved_output: int) -> int:
    """
    Extract actual usage from LangChain/Groq response metadata.
    Fall back to a conservative estimate when unavailable.
    """

    usage = getattr(response, "usage_metadata", None) or {}

    if usage:
        total = usage.get("total_tokens")

        if total is not None:
            return int(total)

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return int(input_tokens + output_tokens)

    metadata = getattr(response, "response_metadata", {}) or {}

    usage = metadata.get("token_usage", {}) or {}

    if usage:
        return int(
            usage.get("total_tokens")
            or (
                usage.get("prompt_tokens", 0)
                + usage.get("completion_tokens", 0)
            )
        )

    # If the provider does not return usage,
    # count the reserved output to avoid undercounting.
    return estimated_input + reserved_output


def budgeted_invoke(
    agent_name: str,
    messages,
    current_usage: int,
    max_output_tokens: int = 500,
):
    """
    Invoke Groq only when the estimated input and reserved
    output fit inside the current travel-query token budget.
    """

    prompt_text = messages_to_text(messages)
    estimated_input = estimate_tokens(prompt_text)

    required_tokens = estimated_input + max_output_tokens
    remaining = TOKEN_LIMIT - current_usage

    print(
        f"\n[TOKEN CHECK] {agent_name}",
        flush=True,
    )

    print(
        f"Current usage: {current_usage}/{TOKEN_LIMIT}",
        flush=True,
    )

    print(
        f"Estimated input: {estimated_input}",
        flush=True,
    )

    print(
        f"Reserved output: {max_output_tokens}",
        flush=True,
    )

    print(
        f"Remaining: {remaining}",
        flush=True,
    )

    if required_tokens > remaining:
        raise RuntimeError(
            f"{agent_name} rejected: token budget insufficient. "
            f"Required approximately {required_tokens}, "
            f"remaining {remaining}."
        )

    response = llm.bind(
        max_tokens=max_output_tokens
    ).invoke(messages)

    actual_tokens = get_actual_token_usage(
        response,
        estimated_input,
        max_output_tokens,
    )

    new_usage = current_usage + actual_tokens

    print(
        f"[TOKEN USAGE] {agent_name}: "
        f"{actual_tokens} tokens | "
        f"Total: {new_usage}/{TOKEN_LIMIT}",
        flush=True,
    )

    if new_usage > TOKEN_LIMIT:
        print(
            "[TOKEN WARNING] Actual usage exceeded the planned "
            "per-query budget. Further calls will be blocked.",
            flush=True,
        )

    return response, new_usage


# =========================================================
# CONTEXT CLEANING AND TRIMMING
# =========================================================

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, str):
        text = value

    elif isinstance(value, Mapping):

        preferred = (
            "name",
            "city",
            "country",
            "iata",
            "icao",
            "airport",
            "airline",
            "price",
            "currency",
            "departure",
            "arrival",
            "duration",
            "date",
            "temperature",
            "description",
            "condition",
            "url",
            "title",
            "content",
            "snippet",
        )

        parts = []

        for key in preferred:
            item = value.get(key)

            if item not in (None, "", [], {}):
                parts.append(
                    f"{key}: {clean_text(item)}"
                )

        text = "\n".join(parts) if parts else str(value)

    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes)
    ):

        text = "\n".join(
            clean_text(item) for item in value
        )

    else:
        text = str(value)

    return re.sub(r"\s+", " ", text).strip()


def limit_context(value: Any, max_chars: int = 1500) -> str:

    text = clean_text(value)

    if not text:
        return "Information unavailable."

    if len(text) <= max_chars:
        return text

    trimmed = text[:max_chars].rsplit(" ", 1)[0]

    return trimmed + " ... [context trimmed]"


def compact_records(
    data: Any,
    relevant_keys: tuple[str, ...],
    max_items: int = 5,
    max_chars: int = 1500,
) -> str:

    if isinstance(data, Mapping):

        records = next(
            (
                value for value in data.values()
                if isinstance(value, list)
            ),
            None,
        )

        if records is None:
            records = [data]

    elif isinstance(data, list):
        records = data

    else:
        return limit_context(data, max_chars)

    output = []

    for record in records[:max_items]:

        if isinstance(record, Mapping):

            fields = []

            for key in relevant_keys:

                value = record.get(key)

                if value not in (None, "", [], {}):
                    fields.append(
                        f"{key}: {clean_text(value)}"
                    )

            if fields:
                output.append(" | ".join(fields))

        else:
            output.append(clean_text(record))

    return limit_context(
        "\n".join(output),
        max_chars,
    )


def build_agent_context(state):

    return {
        "query": limit_context(
            state.get("user_query"),
            800,
        ),

        "constraints": limit_context(
            state.get("trip_constraints"),
            500,
        ),

        "flight": limit_context(
            state.get("flight_results"),
            900,
        ),

        "hotel": limit_context(
            state.get("hotel_results"),
            1000,
        ),

        "weather": limit_context(
            state.get("weather_results"),
            600,
        ),

        "budget": limit_context(
            state.get("budget_results"),
            800,
        ),

        "itinerary": limit_context(
            state.get("itinerary"),
            1800,
        ),
    }


# =========================================================
# STATE SCHEMA
# =========================================================

class GuardrailResult(BaseModel):
    allowed: bool
    reason: str = ""


class TripConstraints(BaseModel):
    destination: str = ""
    origin: str = ""
    duration: str = ""
    budget: str = ""
    travel_style: str = ""
    special_preferences: list[str] = Field(default_factory=list)


class SupervisorResult(BaseModel):
    selected_agents: list[
        Literal[
            "flight_agent",
            "hotel_agent",
            "weather_agent",
            "budget_agent",
            "itinerary_agent",
        ]
    ]
    trip_constraints: TripConstraints
    reasoning: str = ""

class TravelState(TypedDict, total=False):

    messages: Annotated[
        list[AnyMessage],
        operator.add
    ]

    user_query: str

    guardrail_allowed: bool
    guardrail_reason: str

    selected_agents: list[str]
    trip_constraints: TripConstraints
    supervisor_reasoning: str

    flight_results: str
    hotel_results: str
    weather_results: str

    budget_results: str
    itinerary: str

    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int

    # Shared usage counter for this travel request.
    token_usage: int
    token_limit: int




# =========================================================
# SHARED AGENT CONFIGURATION
# =========================================================

KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _empty_constraints():

    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


def _json_from_llm(text: str):

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object returned.")

    return json.loads(text[start:end + 1])


# =========================================================
# SUPERVISOR + GUARDRAIL
# =========================================================


def supervisor_agent(state: TravelState):

    query = limit_context(
        state.get("user_query"),
        800,
    )

    token_usage = state.get("token_usage", 0)
    llm_calls = state.get("llm_calls", 0)

    # -------------------------
    # GUARDRAIL
    # -------------------------

    guardrail_prompt = f"""
Determine whether this request is related to travel.

Valid subjects include:
flights, hotels, weather, budgets, destinations,
visas, transport, sightseeing and itineraries.

Block unrelated requests and harmful or illegal instructions.

User request:
{query}
"""

    try:

        guardrail_messages = [
            SystemMessage(
                content="You are a travel input guardrail."
            ),
            HumanMessage(content=guardrail_prompt),
        ]

        estimated = estimate_tokens(
            messages_to_text(guardrail_messages)
        )

        if not (
            estimated + 150
            <= TOKEN_LIMIT - token_usage
        ):
            raise RuntimeError("Insufficient token budget")

        structured_guardrail = llm.with_structured_output(
            GuardrailResult
        )

        response = structured_guardrail.invoke(
            guardrail_messages
        )

        token_usage += get_actual_token_usage(
            response,
            estimated,
            150,
        )

        llm_calls += 1

        allowed = response.allowed
        reason = response.reason

    except Exception as exc:

        print(
            f"GUARDRAIL ERROR: {exc}",
            flush=True,
        )

        return {
            "guardrail_allowed": False,
            "guardrail_reason": "Unable to validate request.",
            "final_response": "Please retry your travel request.",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "token_usage": token_usage,
            "llm_calls": llm_calls,
        }

    if not allowed:

        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "final_response": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "token_usage": token_usage,
            "llm_calls": llm_calls,
            "messages": [
                AIMessage(content=reason)
            ],
        }

    # -------------------------
    # SUPERVISOR ROUTING
    # -------------------------

    supervisor_prompt = f"""
Select only the specialist agents needed.

Available agents:
- flight_agent: flights and airports
- hotel_agent: accommodation
- weather_agent: weather and forecast
- budget_agent: cost and affordability
- itinerary_agent: integrated travel itinerary

Always include itinerary_agent.

Extract the trip constraints from the user's request.

User request:
{query}
"""

    try:

        supervisor_messages = [
            SystemMessage(
                content=(
                    "You are a travel planning supervisor. "
                    "Select relevant agents and extract trip constraints."
                )
            ),
            HumanMessage(content=supervisor_prompt),
        ]

        estimated = estimate_tokens(
            messages_to_text(supervisor_messages)
        )

        if not (
            estimated + 300
            <= TOKEN_LIMIT - token_usage
        ):
            raise RuntimeError("Insufficient token budget")

        structured_supervisor = llm.with_structured_output(
            SupervisorResult
        )

        response = structured_supervisor.invoke(
            supervisor_messages
        )

        token_usage += get_actual_token_usage(
            response,
            estimated,
            300,
        )

        llm_calls += 1

        selected_agents = [
            agent for agent in AGENT_ORDER
            if agent in response.selected_agents
        ]

        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = response.trip_constraints.model_dump()

        reasoning = response.reasoning

    except Exception as exc:

        print(
            f"SUPERVISOR ERROR: {exc}",
            flush=True,
        )

        return {
            "guardrail_allowed": False,
            "guardrail_reason": (
                "Unable to create the travel agent plan."
            ),
            "final_response": (
                "Please retry with a shorter travel request."
            ),
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "token_usage": token_usage,
            "llm_calls": llm_calls,
        }

    return {
        "guardrail_allowed": True,
        "guardrail_reason": reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "token_usage": token_usage,
        "llm_calls": llm_calls,
        "messages": [
            AIMessage(
                content="Supervisor created the agent plan."
            )
        ],
    }



# =========================================================
# GUARDRAIL BLOCKED
# =========================================================

def guardrail_blocked_agent(state: TravelState):

    reason = state.get("final_response") or (
        state.get("guardrail_reason")
        or "Travel request blocked."
    )

    return {
        "final_response": reason,
        "messages": [
            AIMessage(content=reason)
        ],
    }


# =========================================================
# FLIGHT AGENT
# =========================================================

def flight_agent(state: TravelState):

    print("\nINSIDE FLIGHT AGENT\n", flush=True)

    token_usage = state.get("token_usage", 0)
    llm_calls = state.get("llm_calls", 0)

    constraints = state.get(
        "trip_constraints",
        {},
    )

    origin = constraints.get("origin", "")
    destination = constraints.get("destination", "")

    try:

        airports = run_async(
            aviation_mcp_call("list_airports")
        )

        airlines = run_async(
            aviation_mcp_call("list_airlines")
        )

        airport_data = compact_records(
            airports,
            (
                "name",
                "city",
                "country",
                "iata",
                "icao",
            ),
            max_items=5,
            max_chars=700,
        )

        airline_data = compact_records(
            airlines,
            (
                "name",
                "iata",
                "icao",
            ),
            max_items=5,
            max_chars=400,
        )

        prompt = f"""
User travel request:
{limit_context(state.get('user_query'), 500)}

Origin: {origin}
Destination: {destination}

Airport information:
{airport_data}

Airline information:
{airline_data}

Provide:
1. Likely departure and arrival airports.
2. Flight route guidance.
3. Typical duration only if supported.
4. Approximate airfare only if supported.
5. Booking advice.

Do not claim an airline serves a route based only on
its presence in the airline catalog.
Do not invent live schedules or ticket prices.
"""

        response, token_usage = budgeted_invoke(
            "flight_agent",
            [
                SystemMessage(
                    content="You are an expert flight planner."
                ),
                HumanMessage(content=prompt),
            ],
            token_usage,
            max_output_tokens=500,
        )

        llm_calls += 1

        flight_results = str(response.content)

    except Exception as exc:

        print(
            f"FLIGHT AGENT ERROR: {exc}",
            flush=True,
        )

        flight_results = (
            "Flight information unavailable. "
            "Live schedules and prices must be verified."
        )

    return {
        "flight_results": limit_context(
            flight_results,
            1200,
        ),
        "token_usage": token_usage,
        "llm_calls": llm_calls,
        "messages": [
            AIMessage(
                content="Flight information processed."
            )
        ],
    }


# =========================================================
# HOTEL AGENT
# =========================================================

def hotel_agent(state: TravelState):

    print("\nINSIDE HOTEL AGENT\n", flush=True)

    constraints = state.get(
        "trip_constraints",
        {},
    )

    destination = constraints.get(
        "destination",
        "",
    )

    query = (
        f"Best hotels in {destination}. "
        f"{limit_context(state.get('user_query'), 400)}"
    )

    try:

        raw_results = run_async(
            tavily_mcp_search(query)
        )

        hotel_results = compact_records(
            raw_results,
            (
                "title",
                "name",
                "content",
                "snippet",
                "url",
            ),
            max_items=5,
            max_chars=1400,
        )

    except Exception as exc:

        print(
            f"HOTEL MCP ERROR: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search unavailable. "
            "Hotel availability and prices need verification."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
    }


# =========================================================
# WEATHER AGENT
# =========================================================

def weather_agent(state: TravelState):

    print("\nINSIDE WEATHER AGENT\n", flush=True)

    constraints = state.get(
        "trip_constraints",
        {},
    )

    city = (
        constraints.get("destination")
        or extract_destination(
            state.get("user_query", "")
        )
    )

    try:

        weather_data = run_async(
            weather_mcp_search(city)
        )

        forecast_data = run_async(
            forecast_mcp_search(city)
        )

        weather_results = (
            "Current weather:\n"
            + compact_records(
                weather_data,
                (
                    "city",
                    "temperature",
                    "condition",
                    "description",
                    "humidity",
                ),
                max_items=3,
                max_chars=400,
            )
            + "\nForecast:\n"
            + compact_records(
                forecast_data,
                (
                    "date",
                    "temperature",
                    "condition",
                    "description",
                ),
                max_items=5,
                max_chars=500,
            )
        )

    except Exception as exc:

        print(
            f"WEATHER MCP ERROR: {exc}",
            flush=True,
        )

        weather_results = (
            f"Live weather for {city} unavailable. "
            "Verify the forecast before departure."
        )

    return {
        "weather_results": limit_context(
            weather_results,
            900,
        ),
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }


# =========================================================
# BUDGET AGENT
# =========================================================

def budget_agent(state: TravelState):

    print("\nINSIDE BUDGET AGENT\n", flush=True)

    ctx = build_agent_context(state)

    prompt = f"""
Evaluate the feasibility of this travel budget.

USER REQUEST:
{ctx['query']}

TRIP CONSTRAINTS:
{ctx['constraints']}

FLIGHTS:
{ctx['flight']}

HOTELS:
{ctx['hotel']}

WEATHER:
{ctx['weather']}

Provide:
1. Flight cost estimate.
2. Hotel cost estimate.
3. Food and local transportation.
4. Activities and miscellaneous expenses.
5. Overall estimated budget.
6. Budget risks and savings.

Respect the user's currency.
Never invent live prices.
Clearly distinguish sourced prices from estimates.
Keep the answer concise.
"""

    token_usage = state.get("token_usage", 0)
    llm_calls = state.get("llm_calls", 0)

    try:

        response, token_usage = budgeted_invoke(
            "budget_agent",
            [
                SystemMessage(
                    content="You are a practical travel budget analyst."
                ),
                HumanMessage(content=prompt),
            ],
            token_usage,
            max_output_tokens=650,
        )

        llm_calls += 1

        budget_results = str(response.content)

    except Exception as exc:

        print(
            f"BUDGET AGENT ERROR: {exc}",
            flush=True,
        )

        budget_results = (
            "Budget analysis unavailable within the remaining "
            "token budget. Verify current travel prices."
        )

    return {
        "budget_results": limit_context(
            budget_results,
            1000,
        ),
        "token_usage": token_usage,
        "llm_calls": llm_calls,
        "messages": [
            AIMessage(
                content="Budget assessment generated."
            )
        ],
    }


# =========================================================
# ITINERARY AGENT
# =========================================================


def itinerary_agent(state: TravelState):

    print("\nINSIDE ITINERARY AGENT\n", flush=True)

    ctx = build_agent_context(state)

    # DEBUG: Check the context size passed to the LLM
    print("\n===== ITINERARY CONTEXT DEBUG =====", flush=True)
    print("User query:", len(str(ctx.get("query", ""))), flush=True)
    print("Trip constraints:", len(str(ctx.get("constraints", ""))), flush=True)
    print("Flight results:", len(str(ctx.get("flight", ""))), flush=True)
    print("Hotel results:", len(str(ctx.get("hotel", ""))), flush=True)
    print("Weather results:", len(str(ctx.get("weather", ""))), flush=True)
    print("Budget results:", len(str(ctx.get("budget", ""))), flush=True)

    total_chars = sum(
        len(str(ctx.get(key, "")))
        for key in [
            "query",
            "constraints",
            "flight",
            "hotel",
            "weather",
            "budget",
        ]
    )

    print("Total context characters:", total_chars, flush=True)
    print("Current token usage:", state.get("token_usage", 0), flush=True)
    print("Remaining token budget:",
          state.get("token_limit", 6000) - state.get("token_usage", 0),
          flush=True)
    print("===================================\n", flush=True)

    prompt = f"""
Create a practical, budget-aware travel itinerary.

USER REQUEST:
{ctx['query']}

TRIP CONSTRAINTS:
{ctx['constraints']}

FLIGHTS:
{ctx['flight']}

HOTELS:
{ctx['hotel']}

WEATHER:
{ctx['weather']}

BUDGET:
{ctx['budget']}

Requirements:
- Create a day-by-day itinerary.
- Respect the requested duration and budget.
- Include activities and transportation suggestions.
- Do not invent flight schedules or hotel availability.
- Do not invent live prices.
- Clearly label estimates.
- Keep the itinerary concise and practical.
"""

    token_usage = state.get("token_usage", 0)
    llm_calls = state.get("llm_calls", 0)

    try:

        response, token_usage = budgeted_invoke(
            "itinerary_agent",
            [
                SystemMessage(
                    content=(
                        "You are an expert travel itinerary planner. "
                        "Use the provided evidence and label estimates."
                    )
                ),
                HumanMessage(content=prompt),
            ],
            token_usage,
            max_output_tokens=1200,
        )

        llm_calls += 1

        itinerary = str(response.content)

    except Exception as exc:

        print(
            f"ITINERARY AGENT ERROR: {exc}",
            flush=True,
        )

        itinerary = (
            "The itinerary could not be generated within the "
            "remaining token budget. Please retry with a shorter "
            "request or fewer travel requirements."
        )

    return {
        "itinerary": limit_context(
            itinerary,
            2500,
        ),
        "approval_request": (
            "Review the draft itinerary. Approve it or provide "
            "feedback for revision."
        ),
        "token_usage": token_usage,
        "llm_calls": llm_calls,
        "messages": [
            AIMessage(
                content="Draft itinerary created for human review."
            )
        ],
    }


# =========================================================
# HUMAN APPROVAL
# =========================================================

def human_approval_agent(state: TravelState):

    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get(
                "approval_request",
                "",
            ),
            "selected_agents": state.get(
                "selected_agents",
                [],
            ),
            "supervisor_reasoning": state.get(
                "supervisor_reasoning",
                "",
            ),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(
        review.get("approved", False)
    )

    feedback = str(
        review.get("feedback", "")
    ).strip()

    return {
        "approved": approved,
        "human_feedback": feedback,
        "messages": [
            AIMessage(
                content="Human approval step completed."
            )
        ],
    }


# =========================================================
# FINAL AGENT
# =========================================================

def final_agent(state: TravelState):

    draft = state.get("itinerary", "")

    token_usage = state.get("token_usage", 0)
    llm_calls = state.get("llm_calls", 0)

    # Approved draft: do not spend another Groq call.
    if state.get("approved", False):

        return {
            "final_response": draft,
            "messages": [
                AIMessage(content=draft)
            ],
            "token_usage": token_usage,
            "llm_calls": llm_calls,
        }

    # Rejected draft: send only the draft and feedback.
    prompt = f"""
Revise the existing travel itinerary.

USER REQUEST:
{limit_context(state.get('user_query'), 500)}

CURRENT DRAFT:
{limit_context(draft, 1800)}

HUMAN FEEDBACK:
{limit_context(state.get('human_feedback'), 500)}

Apply the feedback.
Preserve accurate travel details.
Do not invent live prices or availability.
Return the complete revised itinerary concisely.
"""

    try:

        response, token_usage = budgeted_invoke(
            "final_agent",
            [
                SystemMessage(
                    content="You are a professional travel planner."
                ),
                HumanMessage(content=prompt),
            ],
            token_usage,
            max_output_tokens=1000,
        )

        llm_calls += 1

        final_response = str(response.content)

    except Exception as exc:

        print(
            f"FINAL AGENT ERROR: {exc}",
            flush=True,
        )

        # Do not lose the existing draft if the
        # remaining budget cannot support a revision.
        final_response = (
            draft
            + "\n\nRevision unavailable: "
            + "the remaining token budget was insufficient."
        )

    return {
        "final_response": final_response,
        "messages": [
            AIMessage(content=final_response)
        ],
        "token_usage": token_usage,
        "llm_calls": llm_calls,
    }


# =========================================================
# DYNAMIC ROUTING
# =========================================================

ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState):

    selected = state.get("selected_agents", [])

    return [
        agent for agent in AGENT_ORDER
        if agent in selected
    ]


def route_from_supervisor(state: TravelState):

    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)

    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):

    def route(state: TravelState):

        selected = _selected_agents(state)

        current_index = AGENT_ORDER.index(
            current_agent
        )

        for next_agent in AGENT_ORDER[current_index + 1:]:

            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================================================
# BUILD LANGGRAPH
# =========================================================

graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)

graph.add_node(
    "guardrail_blocked",
    guardrail_blocked_agent,
)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")

graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    ROUTE_MAP,
)

graph.add_conditional_edges(
    "flight_agent",
    route_after_agent("flight_agent"),
    ROUTE_MAP,
)

graph.add_conditional_edges(
    "hotel_agent",
    route_after_agent("hotel_agent"),
    ROUTE_MAP,
)

graph.add_conditional_edges(
    "weather_agent",
    route_after_agent("weather_agent"),
    ROUTE_MAP,
)

graph.add_conditional_edges(
    "budget_agent",
    route_after_agent("budget_agent"),
    ROUTE_MAP,
)

graph.add_edge(
    "itinerary_agent",
    "human_approval",
)

graph.add_edge(
    "human_approval",
    "final_agent",
)

graph.add_edge("final_agent", END)

graph.add_edge(
    "guardrail_blocked",
    END,
)


# =========================================================
# POSTGRESQL CHECKPOINTER
# =========================================================

DATABASE_URL = get_database_url()

db_pool = ConnectionPool(
    conninfo=DATABASE_URL,
    max_size=20,
    kwargs={
        "autocommit": True,
        "row_factory": dict_row,
    },
)

checkpointer = PostgresSaver(db_pool)

checkpointer.setup()

travel_graph = graph.compile(
    checkpointer=checkpointer
)


# =========================================================
# FASTAPI SERIALIZATION HELPERS
# =========================================================

def _interrupt_payload(result):

    interrupts = result.get(
        "__interrupt__",
        [],
    )

    if not interrupts:
        return None

    first = interrupts[0]

    payload = getattr(
        first,
        "value",
        first,
    )

    return (
        payload
        if isinstance(payload, dict)
        else {"value": payload}
    )


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
):

    messages = result.get("messages", [])

    last_message = (
        messages[-1].content
        if messages
        else ""
    )

    answer = (
        result.get("final_response")
        or last_message
    )

    interrupt_payload = _interrupt_payload(
        result
    )

    if interrupt_payload:

        answer = (
            interrupt_payload.get("draft_itinerary")
            or result.get("itinerary", "")
        )

    return {
        "thread_id": thread_id,

        "answer": answer,

        "requires_approval": (
            interrupt_payload is not None
        ),

        "approval_request": (
            interrupt_payload.get(
                "approval_request",
                "",
            )
            if interrupt_payload
            else result.get(
                "approval_request",
                "",
            )
        ),

        "flight_results": result.get(
            "flight_results",
            "",
        ),

        "hotel_results": result.get(
            "hotel_results",
            "",
        ),

        "weather_results": result.get(
            "weather_results",
            "",
        ),

        "budget_results": result.get(
            "budget_results",
            "",
        ),

        "itinerary": (
            interrupt_payload.get(
                "draft_itinerary",
                "",
            )
            if interrupt_payload
            else result.get("itinerary", "")
        ),

        "selected_agents": result.get(
            "selected_agents",
            [],
        ),

        "trip_constraints": result.get(
            "trip_constraints",
            {},
        ),

        "supervisor_reasoning": result.get(
            "supervisor_reasoning",
            "",
        ),

        "guardrail_allowed": result.get(
            "guardrail_allowed",
            True,
        ),

        "guardrail_reason": result.get(
            "guardrail_reason",
            "",
        ),

        "approved": result.get("approved"),

        "human_feedback": result.get(
            "human_feedback",
            "",
        ),

        "llm_calls": result.get(
            "llm_calls",
            0,
        ),

        "token_usage": result.get(
            "token_usage",
            0,
        ),

        "token_limit": result.get(
            "token_limit",
            TOKEN_LIMIT,
        ),

        "remaining_tokens": max(
            0,
            TOKEN_LIMIT - result.get(
                "token_usage",
                0,
            ),
        ),
    }


# =========================================================
# START TRAVEL QUERY
# =========================================================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None,
):

    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    initial_state = {

        "messages": [
            HumanMessage(content=user_input)
        ],

        "user_query": user_input,

        "guardrail_allowed": True,
        "guardrail_reason": "",

        "selected_agents": [],

        "trip_constraints": _empty_constraints(),

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

        # Reset token budget for every NEW query.
        "token_usage": 0,
        "token_limit": TOKEN_LIMIT,
    }

    result = travel_graph.invoke(
        initial_state,
        config=config,
    )

    return _serialize_result(
        result,
        thread_id,
    )


# =========================================================
# RESUME AFTER HUMAN APPROVAL
# =========================================================

def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):

    if not thread_id:
        raise ValueError(
            "thread_id is required."
        )

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return (
        _serialize_result(
            result,
            thread_id,
        ),
        thread_id,
    )
