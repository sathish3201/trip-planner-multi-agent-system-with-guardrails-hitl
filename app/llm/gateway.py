
import time
import random
import re

from app.llm.router import router
from app.llm.rate_limiter import HybridTokenRateLimiter
from app.security.prompt_injection import check_prompt_injection
from app.security.pii_guardrail import redact_pii
from app.config import (
    LOCAL_TPM_LIMIT,
    MAX_CONCURRENT_REQUESTS,
    MAX_429_RETRIES,
)


limiter = HybridTokenRateLimiter(
    tpm_limit=LOCAL_TPM_LIMIT,
    max_concurrent=MAX_CONCURRENT_REQUESTS,
)


def sanitize_messages(messages):

    sanitized = []

    for message in messages:

        role = getattr(message, "type", "user")
        content = getattr(message, "content", "")

        if role in ("human", "user"):

            if not isinstance(content, str):
                raise ValueError("Only text messages are supported.")

            check_prompt_injection(content)
            content, detected = redact_pii(content)

            if detected:
                print("[PII REDACTED]", detected)

            role = "user"

        elif role in ("ai", "assistant"):
            role = "assistant"

        elif role == "system":
            role = "system"

        else:
            raise ValueError(f"Unsupported role: {role}")

        sanitized.append({
            "role": role,
            "content": content,
        })

    return sanitized


def estimate_tokens(text):
    return max(1, len(text) // 2)


def get_usage(response, fallback):

    usage = getattr(response, "usage", None)

    if usage is None:
        return fallback

    total = getattr(usage, "total_tokens", None)

    if total:
        return int(total)

    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0

    return int(prompt + completion) or fallback


def is_rate_limit_error(exc):

    text = str(exc).lower()

    return (
        getattr(exc, "status_code", None) == 429
        or "429" in text
        or "rate_limit_exceeded" in text
        or "rate limit reached" in text
    )


def get_retry_after(exc):

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}

    value = headers.get("retry-after")

    if value:
        try:
            return float(value)
        except (ValueError, TypeError):
            pass

    match = re.search(
        r"retry[- ]after[:\s]+([0-9.]+)",
        str(exc),
        re.IGNORECASE,
    )

    return float(match.group(1)) if match else None


def centralized_llm_call(
    messages,
    max_tokens=1000,
    temperature=0.2,
    response_format=None,
):

    messages = sanitize_messages(messages)

    prompt_text = "\n".join(
        item["content"] for item in messages
    )

    estimated = estimate_tokens(prompt_text) + max_tokens

    if estimated > limiter.tpm_limit:
        raise ValueError("Request exceeds local TPM capacity.")

    limiter.semaphore.acquire()

    try:

        for attempt in range(MAX_429_RETRIES + 1):

            reservation = limiter.acquire(estimated)

            try:

                kwargs = {
                    "model": "travel-llm",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                }

                if response_format:
                    kwargs["response_format"] = response_format

                response = router.completion(**kwargs)

                actual = get_usage(response, estimated)

                limiter.settle(reservation, actual)

                return response

            except Exception as exc:

                limiter.release(reservation)

                if not is_rate_limit_error(exc):
                    raise

                if attempt >= MAX_429_RETRIES:
                    raise

                retry_after = get_retry_after(exc)

                delay = (
                    retry_after
                    if retry_after is not None
                    else min(30, 2 ** (attempt + 1))
                )

                delay += random.uniform(0.1, 1.0)

                print(f"[LLM RETRY] Waiting {delay:.2f}s")

                time.sleep(delay)

    finally:
        limiter.semaphore.release()
