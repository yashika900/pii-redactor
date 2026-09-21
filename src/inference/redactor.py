"""
Core PIIRedactor class: combines a locally-served fine-tuned language model
with a regex-based safety net to redact PII from text, entirely offline.
"""

import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Tuple

from .regex_patterns import apply_regex_redaction

SYSTEM_PROMPT = (
    "You are a PII redaction assistant. Given a piece of text, identify and "
    "replace all personally identifiable information with the appropriate "
    "tags (e.g., [NAME], [EMAIL], [PHONE], [ADDRESS], [DOB], [SSN], "
    "[CREDIT_CARD], [BANK_ACCOUNT], [PASSPORT], [DRIVER_LICENSE], "
    "[IP_ADDRESS], [USERNAME], [PASSWORD], [URL], [ORG], [DATE], "
    "[LICENSE_PLATE], [MEDICAL_ID], [EMPLOYEE_ID], [AGE]). If there is no "
    "personal information, return the text unchanged."
)


@dataclass
class RedactionResult:
    """Structured result of a redaction call."""

    original: str
    model_output: str
    final_output: str
    regex_safety_net_catches: List[Tuple[str, int]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def safety_net_triggered(self) -> bool:
        """True if the regex layer caught PII the model itself missed."""
        return len(self.regex_safety_net_catches) > 0


class PIIRedactor:
    """
    Two-layer PII redactor:
        1. A locally-served, fine-tuned small language model (via Ollama)
           handles contextual PII (names, addresses, indirect references).
        2. A regex safety net catches structurally predictable PII
           (emails, phone numbers, SSNs, IPs, etc.) that the model may
           occasionally miss, particularly on long or complex inputs.

    All inference happens on-device via Ollama; no network calls are made
    at redaction time.
    """

    def __init__(self, model_name: str = "pii-redactor", timeout: int = 60):
        self.model_name = model_name
        self.timeout = timeout

    def _run_model(self, text: str) -> str:
        """Call the local Ollama model and return its raw text output."""
        result = subprocess.run(
            ["ollama", "run", self.model_name, text],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Ollama call failed (exit code {result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return result.stdout.strip()

    def redact(self, text: str) -> RedactionResult:
        """
        Redact PII from the given text using the full two-layer pipeline.

        Args:
            text: Raw input text potentially containing PII.

        Returns:
            A RedactionResult with the original text, the model's raw
            output, the final redacted output (after the regex safety
            net), and a log of anything the safety net additionally caught.
        """
        start = time.time()
        model_output = self._run_model(text)
        final_output, regex_catches = apply_regex_redaction(model_output)
        elapsed = time.time() - start

        return RedactionResult(
            original=text,
            model_output=model_output,
            final_output=final_output,
            regex_safety_net_catches=regex_catches,
            elapsed_seconds=elapsed,
        )
