"""
Regex-based PII detection patterns.

Serves as a safety-net layer applied after the fine-tuned model's output,
catching structurally predictable PII (emails, phone numbers, IDs, etc.)
that the model may occasionally miss on longer or more complex inputs.

Patterns are ordered from most specific/context-anchored to least specific,
so that more precise patterns claim their matches before looser ones run.
"""

import re
from typing import Dict, List, Tuple

# Order matters: specific/context-anchored patterns run first to avoid
# looser patterns (e.g. PHONE) accidentally consuming digits that belong
# to a different PII type (e.g. an account number or credit card).
REGEX_PATTERNS: Dict[str, re.Pattern] = {
    "DOB": re.compile(
        r"\b(?:DOB|Date of Birth|born(?: on)?)[:\s]+\d{4}-\d{2}-\d{2}\b|"
        r"\b(?:DOB|Date of Birth|born(?: on)?)[:\s]+\d{1,2}/\d{1,2}/\d{2,4}\b",
        re.IGNORECASE,
    ),
    "BANK_ACCOUNT": re.compile(
        r"\b(?:account|acct)(?:\s*(?:number|#|no\.?))?[:\s]+\d{8,17}\b",
        re.IGNORECASE,
    ),
    "PASSPORT": re.compile(
        r"\b(?:passport)(?:\s*(?:number|#|no\.?))?[:\s]*[A-Z]{1,2}\d{6,9}\b",
        re.IGNORECASE,
    ),
    "DRIVER_LICENSE": re.compile(
        r"\b(?:driver.?s?\s*licen[cs]e)(?:\s*(?:number|#|no\.?))?[:\s]*[A-Z0-9]{6,12}\b",
        re.IGNORECASE,
    ),
    "CREDIT_CARD": re.compile(
        r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b"
    ),
    "SSN": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "IP_ADDRESS": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "URL": re.compile(
        r"\bhttps?://[^\s<>\"']+|www\.[^\s<>\"']+\b"
    ),
    "DATE": re.compile(
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    ),
    "AGE": re.compile(
        r"\b(?:age[d]?[:\s]+)\d{1,3}\b|\b\d{1,3}[-\s]years?[-\s]old\b",
        re.IGNORECASE,
    ),
    # Requires "license plate"/"plate" context to avoid false positives on
    # reference/ticket numbers that share a similar letter-digit shape
    # (e.g. "REF-9012").
    "LICENSE_PLATE": re.compile(
        r"\b(?:license plate|plate number|plate)[:\s]*[A-Z]{1,3}[-\s]?\d{3,4}[-\s]?[A-Z]{0,3}\b",
        re.IGNORECASE,
    ),
    # Requires phone-like grouping (3-3-4 digits with separators) so it
    # doesn't swallow arbitrary long digit runs like account numbers.
    "PHONE": re.compile(
        r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
    ),
}


def apply_regex_redaction(text: str) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Apply all regex patterns to the given text, replacing matches with
    their corresponding [TAG] placeholder.

    Args:
        text: The text to scan and redact (typically the model's output,
            used as a safety net for anything the model missed).

    Returns:
        A tuple of:
            - The redacted text.
            - A list of (tag, match_count) pairs for every pattern that
              matched at least once, useful for logging/auditing what the
              safety net caught.
    """
    redacted_text = text
    matches_found: List[Tuple[str, int]] = []

    for tag, pattern in REGEX_PATTERNS.items():
        matches = pattern.findall(redacted_text)
        if matches:
            matches_found.append((tag, len(matches)))
        redacted_text = pattern.sub(f"[{tag}]", redacted_text)

    return redacted_text, matches_found
