
import re

from collections.abc import Mapping, Sequence


def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, str):
        text = value

    elif isinstance(value, Mapping):

        preferred = (
            "name", "city", "country", "iata",
            "airport", "airline", "price",
            "currency", "departure", "arrival",
            "duration", "date", "temperature",
            "condition", "description", "url",
            "title", "content", "snippet",
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

        text = "\n".join(clean_text(item) for item in value)

    else:
        text = str(value)

    return re.sub(r"\s+", " ", text).strip()


def limit_context(value, max_chars=1500):

    text = clean_text(value)

    if not text:
        return "Information unavailable."

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0] + " ... [trimmed]"


def compact_records(
    data,
    relevant_keys,
    max_items=5,
    max_chars=1500,
):

    if isinstance(data, Mapping):

        records = next(
            (
                value for value in data.values()
                if isinstance(value, list)
            ),
            None,
        )

        records = records if records is not None else [data]

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

    effective_query = (
        state.get("active_query")
        or state.get("user_query")
    )

    return {
        "query": limit_context(
            effective_query, 900
        ),

        "feedback": limit_context(
            state.get("human_feedback"), 400
        ),

        "revision_count": state.get("revision_count", 0),

        "constraints": limit_context(
            state.get("trip_constraints"), 650
        ),

        "flight": limit_context(
            state.get("flight_results"), 1400
        ),

        "hotel": limit_context(
            state.get("hotel_results"), 1450
        ),

        "weather": limit_context(
            state.get("weather_results"), 800
        ),

        "budget": limit_context(
            state.get("budget_results"), 1150
        ),

        "itinerary": limit_context(
            state.get("itinerary"), 1850
        ),
    }