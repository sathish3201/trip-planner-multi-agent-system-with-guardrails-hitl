from app.config import MAX_REVISIONS

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]

ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
    "human_approval": "human_approval",
    "final_agent": "final_agent",
}


def selected_agents(state):

    selected = state.get("selected_agents", [])

    return [
        agent
        for agent in AGENT_ORDER
        if agent in selected
    ]


def route_from_supervisor(state):

    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = selected_agents(state)

    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent):

    def route(state):

        selected = selected_agents(state)

        index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[index + 1:]:

            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route





def route_after_human_approval(state) -> str:
    if state.get("approved"):
        return "final_agent"

    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "final_agent"

    revision_count = state.get("revision_count", 0) + 1
    state["revision_count"] = revision_count
    state['prev_itinerary'] = state.get("itinerary", "")
    state['itinerary'] = ""
    print('returning to supervisor for revision', state['prev_itinerary'], state['itinerary'], state['revision_count'])

    return "supervisor"