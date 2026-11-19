
import re


INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) "
    r"(instructions?|prompts?|rules?)",

    r"disregard (the |all )?(previous|prior|earlier)",

    r"forget (everything|your instructions?|the rules?)",

    r"you are (now |a )?(DAN|jailbroken|unrestricted|unfiltered)",

    r"pretend (you are|to be) .{0,40}"
    r"(no restrictions?|uncensored)",

    r"</?(system|user|assistant|im_start|im_end)>",

    r"new (instructions?|system prompt|rules?):",

    r"reveal your (system )?prompt",

    r"what (are|were) your (original )?instructions?",
]

INJECTION_REGEX = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in INJECTION_PATTERNS
]


class GuardrailViolation(Exception):
    pass


def check_prompt_injection(text: str) -> None:

    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    for regex in INJECTION_REGEX:

        if regex.search(text):

            print(
                "[SECURITY] Prompt injection detected.",
                flush=True
            )

            raise GuardrailViolation(
                "Request blocked by prompt-injection guardrail."
            )
