import re
import litellm
from litellm import completion


PII_PATTERNS = {
    "EMAIL":       r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE_IN":    r"(\+91[\-\s]?)?[6-9]\d{9}",                  # Indian mobile
    "PHONE_US":    r"(\+1[\-\s]?)?\(?\d{3}\)?[\-\s]?\d{3}[\-\s]?\d{4}",
    "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
    "AADHAAR":     r"\b\d{4}\s?\d{4}\s?\d{4}\b",                 # Indian Aadhaar
    "PAN":         r"\b[A-Z]{5}\d{4}[A-Z]\b",                    # Indian PAN
    "CREDIT_CARD": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "IP_ADDRESS":  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


def redact_pii(text: str):
    """Replace PII in text with placeholders. Returns (clean_text, detected_list)."""
    detected = []
    clean = text
    for label, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, clean)
        if matches:
            detected.append({"type": label, "count": len(matches)})
            clean = re.sub(pattern, f"<{label}_REDACTED>", clean)
    return clean, detected


def pii_input_guardrail(kwargs):
    """LiteLLM pre-call hook: scrub PII from user messages."""
    messages = kwargs.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            clean, detected = redact_pii(msg["content"])
            if detected:
                print(f"🚨 PII REDACTED: {detected}")
                msg["content"] = clean


# litellm.input_callback = [pii_input_guardrail]

INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) (instructions?|prompts?|rules?)",
    r"disregard (the |all )?(previous|prior|earlier)",
    r"forget (everything|your instructions?|the rules?)",
    r"you are (now |a )?(DAN|jailbroken|unrestricted|unfiltered)",
    r"pretend (you are|to be) .{0,40}(no restrictions?|uncensored)",
    r"</?(system|user|assistant|im_start|im_end)>",
    r"new (instructions?|system prompt|rules?):",
    r"reveal your (system )?prompt",
    r"what (are|were) your (original )?instructions?",
]

INJECTION_REGEX = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


class GuardrailViolation(Exception):
    """Raised when a guardrail blocks a request."""
    pass


def injection_guardrail(kwargs):
    messages = kwargs.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            content = msg["content"]
            for regex in INJECTION_REGEX:
                if regex.search(content):
                    print(f"🚨 PROMPT INJECTION DETECTED — pattern: {regex.pattern!r}")
                    raise GuardrailViolation("Blocked: prompt injection attempt")



# litellm.input_callback = [injection_guardrail]


# Keywords your assistant should refuse to discuss
FORBIDDEN_TOPICS = [
    "weapon", "bomb", "explosive",
    "hack", "exploit", "malware",
    "drugs", "illegal substance",
    "self-harm", "suicide",
]


class GuardrailViolation(Exception):
    pass


def topic_guardrail(kwargs):
    messages = kwargs.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            content_lower = msg["content"].lower()
            for keyword in FORBIDDEN_TOPICS:
                if keyword in content_lower:
                    print(f"🚨 FORBIDDEN TOPIC: '{keyword}' detected")
                    raise GuardrailViolation(
                        f"This assistant doesn't discuss topics related to '{keyword}'."
                    )


litellm.input_callback = [pii_input_guardrail, injection_guardrail, topic_guardrail]

## small routing ##########

import time
from litellm import completion, completion_cost

def classify_task(user_query: str) -> str:
    """Cheap classifier — uses the fastest model to decide routing."""
    cls = completion(
        model="groq/llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": (
                f"Classify the following query into EXACTLY one word: "
                f"'code', 'summary', or 'general'. Query: {user_query}\n\nAnswer:"
            )
        }],
        max_tokens=5
    )
    return cls.choices[0].message.content.strip().lower()


def call_with_fallbacks(model_chain, messages):
    """Try each model in order; return the first one that succeeds."""
    last_error = None
    for model in model_chain:
        try:
            return completion(model=model, messages=messages)
        except Exception as e:
            print(f"   ⚠️  {model} failed ({type(e).__name__}), trying next...")
            last_error = e
            continue
    raise last_error


def smart_chat(user_query: str):
    """Routes to the right model based on task type, with fallbacks."""
    task = classify_task(user_query)

    # Each entry is a FULL chain: [primary, fallback1, fallback2, ...]
    # Every model name includes its provider prefix (groq/, anthropic/, etc.)
    MODEL_ROUTING = {
    "code": [
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
        "groq/qwen/qwen3.8-27b",
    ],
    "summary": [
        "groq/openai/gpt-oss-20b",
        "groq/qwen/qwen3.8-27b",
    ],
    "general": [
        "groq/openai/gpt-oss-20b",
        "groq/openai/gpt-oss-120b",
    ],
    }
    model_chain = MODEL_ROUTING.get(task, MODEL_ROUTING["general"])

    start = time.time()
    response = call_with_fallbacks(
        model_chain=model_chain,
        messages=[{"role": "user", "content": user_query}]
    )
    latency = time.time() - start

    try:
        cost = completion_cost(completion_response=response)
        cost_str = f"${cost:.6f}"
    except Exception:
        cost_str = "n/a"

    return {
        "detected_task": task,
        "model_used":    response.model,
        "answer":        response.choices[0].message.content,
        "latency_sec":   round(latency, 2),
        "cost_usd":      cost_str
    }




