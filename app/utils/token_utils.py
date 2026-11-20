
from collections.abc import Mapping

from langchain_core.messages import AIMessage

from app.llm.gateway import centralized_llm_call
from app.config import TOKEN_LIMIT


def estimate_tokens(text):

    return max(1, len(text) // 2)


def messages_to_text(messages):

    return "\n".join(
        str(getattr(message, "content", ""))
        for message in messages
    )


def usage_value(usage, key, default=0):

    if usage is None:
        return default

    if isinstance(usage, Mapping):
        return usage.get(key, default)

    return getattr(usage, key, default)


def budgeted_invoke(
    agent_name,
    messages,
    current_usage,
    max_output_tokens=500,
    response_format=None,
    temperature=0.2,
):

    estimated_input = estimate_tokens(
        messages_to_text(messages)
    )

    required = estimated_input + max_output_tokens

    remaining = TOKEN_LIMIT - current_usage

    # if required > remaining:
    #     raise RuntimeError(
    #         f"{agent_name}: Insufficient token budget."
    #     )

    response = centralized_llm_call(
        messages=messages,
        max_tokens=max_output_tokens,
        temperature=temperature,
        response_format=response_format,
    )

    choices = getattr(response, "choices", None) or []

    if not choices:
        raise ValueError("LLM returned no choices.")

    message = choices[0].message

    content = getattr(message, "content", None)

    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned empty content.")

    usage = getattr(response, "usage", None)

    input_tokens = usage_value(
        usage,
        "prompt_tokens",
        estimated_input,
    )

    output_tokens = usage_value(
        usage,
        "completion_tokens",
        max_output_tokens,
    )

    total = usage_value(
        usage,
        "total_tokens",
        int(input_tokens) + int(output_tokens),
    )

    total = int(total)

    ai_message = AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": total,
        },
    )

    return ai_message, current_usage + total
