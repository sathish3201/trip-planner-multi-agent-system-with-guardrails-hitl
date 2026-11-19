import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage

from app.utils.telemetry import merge_agent_metrics


class TravelState(TypedDict, total=False):

    messages: Annotated[
        list[AnyMessage],
        operator.add,
    ]

    user_query: str

    guardrail_allowed: bool
    guardrail_reason: str

    selected_agents: list[str]
    trip_constraints: dict
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
    token_usage: int
    token_limit: int

    agent_metrics: Annotated[
        dict[str, dict[str, Any]],
        merge_agent_metrics,
    ]
    overall_latency_ms: float
    total_cost_usd: float
    total_cost_inr: float
    overall_accuracy: float