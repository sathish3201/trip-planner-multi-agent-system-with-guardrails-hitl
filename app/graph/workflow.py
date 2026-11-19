
from langgraph.graph import StateGraph, START, END

from app.graph.state import TravelState
from app.graph.routing import (
    route_from_supervisor,
    route_after_agent,
    ROUTE_MAP,
)

from app.agents import (
    supervisor_agent,
    guardrail_blocked_agent,
    flight_agent,
    hotel_agent,
    weather_agent,
    budget_agent,
    itinerary_agent,
    human_approval_agent,
    final_agent,
)


def build_workflow(checkpointer):

    graph = StateGraph(TravelState)

    # Register nodes.
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("guardrail_blocked", guardrail_blocked_agent)

    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("weather_agent", weather_agent)
    graph.add_node("budget_agent", budget_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    

    graph.add_node("human_approval", human_approval_agent)
    graph.add_node("final_agent", final_agent)

    # Starting node.
    graph.add_edge(START, "supervisor")

    # Supervisor routing.
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ROUTE_MAP,
    )

    # Dynamic agent routing.
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

    # Human approval.
    graph.add_edge(
        "itinerary_agent",
        "human_approval",
    )

    graph.add_edge(
        "human_approval",
        "final_agent",
    )

    graph.add_edge(
        "final_agent",
        END,
    )

    graph.add_edge(
        "guardrail_blocked",
        END,
    )

    return graph.compile(
        checkpointer=checkpointer
    )
