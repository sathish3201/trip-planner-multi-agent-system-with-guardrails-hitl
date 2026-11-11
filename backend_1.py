import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import nest_asyncio

# Enable nested event loops for sync functions inside FastAPI
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
# from langchain_ollama import ChatOllama

from mcp_client import (
    _require_env,
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# =========================
# Async Helper Function
# =========================
def run_async(coro):
    """Safely execute async coroutines from synchronous graph node functions."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)

def _require_env(name: str, value: str | None) -> str:
    """Return an environment value or raise a readable setup error."""

    if not value:
        raise RuntimeError(
            f"{name} is missing. "
            f"Add {name}=your_key to the project .env file."
        )

    return value

# llm = ChatOllama(model="qwen2.5-coder:1.5b", base_url="http://localhost:11434")
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=_require_env("GROQ_API_KEY", GROQ_API_KEY),
)


def limit_context(value: Any, max_chars: int = 2500) -> str:
    """Keep agent context within a manageable size."""
    if not value:
        return "Information unavailable."

    text = str(value)

    if len(text) > max_chars:
        return text[:max_chars] + "\n[Context truncated]"

    return text
    
# =========================
# State Schema
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


# =========================
# Shared helpers
# =========================
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


import re
from collections.abc import Mapping, Sequence

# Maximum character budgets for each agent's context.
CONTEXT_LIMITS = {
    "user_query": 1000,
    "constraints": 700,
    "flight": 1200,
    "hotel": 1400,
    "weather": 800,
    "budget": 1000,
    "itinerary": 2500,
}


def clean_text(value: Any) -> str:
    """Convert MCP/model output into clean, readable text."""

    if value is None:
        return ""

    if isinstance(value, str):
        text = value
    elif isinstance(value, Mapping):
        # Keep useful fields instead of dumping the entire object.
        preferred = (
            "name", "city", "country", "iata", "icao",
            "airport", "airline", "price", "currency",
            "departure", "arrival", "duration", "date",
            "temperature", "description", "condition",
            "url", "title", "content", "snippet",
        )

        parts = []
        for key in preferred:
            item = value.get(key)
            if item not in (None, "", [], {}):
                parts.append(f"{key}: {clean_text(item)}")

        text = "\n".join(parts) if parts else str(value)

    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        text = "\n".join(clean_text(item) for item in value)

    else:
        text = str(value)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def trim_text(value: Any, max_chars: int) -> str:
    """Trim text at a word boundary."""

    text = clean_text(value)

    if not text:
        return "Information unavailable."

    if len(text) <= max_chars:
        return text

    trimmed = text[:max_chars].rsplit(" ", 1)[0]
    return trimmed + " ... [trimmed]"


def compact_records(
    data: Any,
    relevant_keys: tuple[str, ...],
    max_items: int = 5,
    max_chars: int = 1500,
) -> str:
    """
    Extract selected fields from list/dictionary MCP results.
    Useful for search results and structured API responses.
    """

    if isinstance(data, Mapping):
        # Find a list of records inside a wrapper response.
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
        return trim_text(data, max_chars)

    output = []

    for record in records[:max_items]:
        if isinstance(record, Mapping):
            fields = []
            for key in relevant_keys:
                value = record.get(key)
                if value not in (None, "", [], {}):
                    fields.append(f"{key}: {clean_text(value)}")

            if fields:
                output.append(" | ".join(fields))
        else:
            output.append(clean_text(record))

    return trim_text("\n".join(output), max_chars)


def build_agent_context(state: TravelState) -> dict[str, str]:
    """Build a small, reusable context package for downstream agents."""

    return {
        "user_query": trim_text(
            state.get("user_query"), CONTEXT_LIMITS["user_query"]
        ),
        "constraints": trim_text(
            state.get("trip_constraints"), CONTEXT_LIMITS["constraints"]
        ),
        "flight": trim_text(
            state.get("flight_results"), CONTEXT_LIMITS["flight"]
        ),
        "hotel": trim_text(
            state.get("hotel_results"), CONTEXT_LIMITS["hotel"]
        ),
        "weather": trim_text(
            state.get("weather_results"), CONTEXT_LIMITS["weather"]
        ),
        "budget": trim_text(
            state.get("budget_results"), CONTEXT_LIMITS["budget"]
        ),
        "itinerary": trim_text(
            state.get("itinerary"), CONTEXT_LIMITS["itinerary"]
        ),
    }



def filter_relevant_records(
    data: Any,
    search_terms: list[str],
    max_items: int = 5,
    max_chars: int = 1200,
) -> str:
    """Rank records by whether they mention the requested route."""

    if isinstance(data, Mapping):
        records = next(
            (v for v in data.values() if isinstance(v, list)),
            None,
        )
        if records is None:
            records = [data]
    elif isinstance(data, list):
        records = data
    else:
        return trim_text(data, max_chars)

    terms = [term.lower() for term in search_terms if term]
    ranked = []

    for record in records:
        text = clean_text(record)
        lowered = text.lower()
        score = sum(term in lowered for term in terms)

        if score > 0:
            ranked.append((score, text))

    ranked.sort(key=lambda item: item[0], reverse=True)

    selected = [text for _, text in ranked[:max_items]]

    if not selected:
        # Do not pretend arbitrary catalog entries are route matches.
        return "No matching route records found in the available catalog."

    return trim_text("\n".join(selected), max_chars)


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)




def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent
# =========================
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")
    query = state["user_query"]

    try:
        airports = run_async(aviation_mcp_call("list_airports"))
        airlines = run_async(aviation_mcp_call("list_airlines"))

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt),
            ]
        )
        flight_data = response.content
    except Exception as exc:
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent
# =========================
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"

    try:
        # hotel_results = run_async(tavily_mcp_search(query))
        hotel_results = limit_context(
    run_async(tavily_mcp_search(query)),
    max_chars=2500
)
    except Exception as exc:
        print(f"HOTEL AGENT MCP ERROR: {type(exc).__name__}: {exc}", flush=True)
        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information processed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Weather Agent
# =========================
def weather_agent(state: TravelState):
    # city = extract_destination(state["user_query"])
    city = (
    state.get("trip_constraints", {}).get("destination")
    or extract_destination(state["user_query"])
)
    try:
        weather_data = run_async(weather_mcp_search(city))
        forecast_data = run_async(forecast_mcp_search(city))

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""
    except Exception as exc:
        print(f"WEATHER AGENT MCP ERROR: {type(exc).__name__}: {exc}", flush=True)
        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather information processed.")],
    }


# =========================
# Budget Agent
# =========================
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent
# =========================
# def itinerary_agent(state: TravelState):
#     prompt = f"""
# Create a complete travel itinerary.

# User Query:
# {state['user_query']}

# Trip Constraints:
# {state.get('trip_constraints', {})}

# Flight Results:
# {state.get('flight_results', '')}

# Hotel Results:
# {state.get('hotel_results', '')}

# Weather Results:
# {state.get('weather_results', '')}

# Budget Results:
# {state.get('budget_results', '')}

# Make the itinerary practical, budget-aware, and easy to follow.
# Create a clear draft that is ready for human review.
# """

#     response = llm.invoke(
#         [
#             SystemMessage(content="You are an expert travel planner."),
#             HumanMessage(content=prompt),
#         ]
#     )

#     approval_request = (
#         "Please review the generated draft itinerary. Approve it to create the "
#         "final polished plan, or provide feedback for revision."
#     )

#     return {
#         "itinerary": response.content,
#         "approval_request": approval_request,
#         "messages": [AIMessage(content="Draft itinerary created for human review.")],
#         "llm_calls": state.get("llm_calls", 0) + 1,
#     }


def itinerary_agent(state: TravelState):
    print("\nINSIDE ITINERARY AGENT\n")

    prompt = f"""
Create a practical travel itinerary for the user.

USER REQUEST:
{limit_context(state.get("user_query"), 1000)}

TRIP CONSTRAINTS:
{limit_context(state.get("trip_constraints"), 700)}

FLIGHT INFORMATION:
{limit_context(state.get("flight_results"), 1200)}

HOTEL INFORMATION:
{limit_context(state.get("hotel_results"), 1800)}

WEATHER INFORMATION:
{limit_context(state.get("weather_results"), 1000)}

BUDGET INFORMATION:
{limit_context(state.get("budget_results"), 1200)}

Instructions:
- Create a day-by-day itinerary.
- Consider the user's budget and travel preferences.
- Do not invent live prices or availability.
- Clearly label estimates.
- Keep the itinerary concise and practical.
- Include activities, transport suggestions, and useful travel tips.
"""

    response = llm.bind(max_tokens=1200).invoke(
        [
            SystemMessage(
                content="You are an expert travel itinerary planner."
            ),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "itinerary": response.content,
        "approval_request": (
            "Please review the generated draft itinerary. "
            "Approve it or provide feedback for revision."
        ),
        "messages": [
            AIMessage(content="Draft itinerary created for human review.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent
# =========================
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
    Generate the final travel response.

    USER REQUEST:
    {limit_context(state.get("user_query"), 1000)}

    HUMAN REVIEW:
    {limit_context(review_instruction, 500)}

    TRIP CONSTRAINTS:
    {limit_context(state.get("trip_constraints"), 700)}

    FLIGHTS:
    {limit_context(state.get("flight_results"), 1000)}

    HOTELS:
    {limit_context(state.get("hotel_results"), 1500)}

    WEATHER:
    {limit_context(state.get("weather_results"), 800)}

    BUDGET:
    {limit_context(state.get("budget_results"), 1000)}

    DRAFT ITINERARY:
    {limit_context(state.get("itinerary"), 2500)}

    Create a concise final travel plan with:
    1. Trip summary
    2. Flights
    3. Hotels
    4. Weather
    5. Day-by-day itinerary
    6. Estimated budget
    7. Recommendations

    Do not invent live prices or availability.
    Important:
    - Be clear and practical.
    - Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
    - Include weather-based travel advice.
    - Keep the response useful for real travel planning.
    - Incorporate the human feedback when revision was requested.
    """
    response = llm.bind(max_tokens=1500).invoke(
            [
                SystemMessage(
                    content="You are a professional AI travel planning assistant."
                ),
                HumanMessage(content=final_prompt),
            ]
        )
    return {
            "final_response": response.content,
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }
    
#     final_prompt = f"""
# Generate the final travel response for the user.

# Human Review:
# {review_instruction}

# User Request:
# {state['user_query']}

# Supervisor Constraints:
# {state.get('trip_constraints', {})}

# Flights:
# {state.get('flight_results', '')}

# Hotels:
# {state.get('hotel_results', '')}

# Weather:
# {state.get('weather_results', '')}

# Budget Analysis:
# {state.get('budget_results', '')}

# Draft Itinerary:
# {state.get('itinerary', '')}

# Format the final answer beautifully using these sections:
# 1. Trip Summary
# 2. Flight Information
# 3. Hotel Suggestions
# 4. Weather Information
# 5. Day-by-Day Itinerary
# 6. Estimated Budget
# 7. Final Recommendations

# Important:
# - Be clear and practical.
# - Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
# - Include weather-based travel advice.
# - Keep the response useful for real travel planning.
# - Incorporate the human feedback when revision was requested.
# """

#     response = llm.invoke(
#         [
#             SystemMessage(
#                 content="You are a professional AI travel booking assistant."
#             ),
#             HumanMessage(content=final_prompt),
#         ]
#     )

#     return {
#         "final_response": response.content,
#         "messages": [response],
#         "llm_calls": state.get("llm_calls", 0) + 1,
#     }



# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)


# =========================
# Thread-safe PostgreSQL Connection Pool
# =========================
DATABASE_URL = get_database_url()

db_pool = ConnectionPool(
    conninfo=DATABASE_URL,
    max_size=20,
    kwargs={"autocommit": True, "row_factory": dict_row},
)

checkpointer = PostgresSaver(db_pool)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI Helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
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
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)