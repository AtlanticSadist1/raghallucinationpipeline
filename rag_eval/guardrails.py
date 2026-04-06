from __future__ import annotations

import re
from typing import Tuple

GUARDRAIL_CHECKS: list[tuple[str, re.Pattern]] = [
    (
        "sensitive personal data",
        re.compile(
            r"\b(?:ssn|social security number|credit card|cvv|bank account|routing number|password|passphrase|private key|secret key|api key|seed phrase)\b",
            re.I,
        ),
    ),
    (
        "illegal or harmful action",
        re.compile(
            r"\b(?:hack|hack into|attack|bomb|murder|poison|terrorist|terrorism|drug lab|drug trafficking|sell drugs|exploit|ransomware|malware|virus|spyware|trojan)\b",
            re.I,
        ),
    ),
    (
        "self-harm or violence",
        re.compile(
            r"\b(?:suicide|self harm|hurt myself|kill myself|cut myself|rape|torture|kill someone|murder someone|bloodbath|violent act)\b",
            re.I,
        ),
    ),
    (
        "hate or harassment",
        re.compile(
            r"\b(?:hate|exterminate|ethnic cleansing|nazi|kkk|racist|racism|slur|insult|degrade)\b",
            re.I,
        ),
    ),
    (
        "prompt injection or system override",
        re.compile(
            r"(ignore|disregard|forget|override|disobey).{0,80}?previous|prior|above|earlier|system prompt|developer instructions|do not obey|follow only|you are not|you are now|output only",
            re.I,
        ),
    ),
]


def guard_text(text: str) -> Tuple[bool, str | None]:
    """Return whether a piece of text is safe to process, and a rejection reason if not."""
    if not text or not text.strip():
        return True, None

    for category, pattern in GUARDRAIL_CHECKS:
        if pattern.search(text):
            return False, (
                "This request cannot be processed for safety reasons. "
                f"It appears to contain {category} that is not permitted."
            )

    return True, None
