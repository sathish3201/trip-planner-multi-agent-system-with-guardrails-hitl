
import operator

from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage


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
