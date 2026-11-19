
import re


PII_PATTERNS = {
    "EMAIL": (
        r"\b[a-zA-Z0-9._%+-]+"
        r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    ),

    "PHONE_IN": (
        r"(?<!\d)(?:\+91[\s-]?)?"
        r"[6-9]\d{9}(?!\d)"
    ),

    "PHONE_US": (
        r"(?<!\d)(?:\+1[\s-]?)?"
        r"(?:\(\d{3}\)|\d{3})"
        r"[\s-]?\d{3}[\s-]?\d{4}(?!\d)"
    ),

    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",

    "AADHAAR": (
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    ),

    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",

    "CREDIT_CARD": (
        r"\b(?:\d{4}[\s-]?){3}\d{4}\b"
    ),
}


def redact_pii(text: str):

    if not isinstance(text, str):
        return text, []

    clean = text
    detected = []

    for label, pattern in PII_PATTERNS.items():

        matches = re.findall(pattern, clean)

        if matches:

            detected.append({
                "type": label,
                "count": len(matches)
            })

            clean = re.sub(
                pattern,
                f"<{label}_REDACTED>",
                clean
            )

    return clean, detected
