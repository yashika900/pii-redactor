"""
Cloud-deployable PII redactor using llama-cpp-python directly, instead of
Ollama. Used for the Streamlit Community Cloud deployment, where system-
level installs (like Ollama) aren't available, but a pure-Python inference
library is.

Functionally equivalent to inference.redactor.PIIRedactor (same model,
same two-layer architecture) — this class exists purely because of the
hosting environment's constraints, not a difference in approach.
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

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

HF_REPO_ID = "Yashika900/pii-redactor-llama3.2-3b"
HF_FILENAME = "pii_redactor_q4.gguf"
LOCAL_MODEL_DIR = "models"


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
        return len(self.regex_safety_net_catches) > 0


class PIIRedactorCloud:
    """
    Two-layer PII redactor for cloud environments without Ollama.

    On first use, downloads the quantized GGUF model from the Hugging Face
    Hub (cached locally afterward) and loads it directly via
    llama-cpp-python, rather than delegating to a separately-running
    Ollama server.
    """

    def __init__(self, n_ctx: int = 4096, repeat_penalty: float = 1.1):
        model_path = self._ensure_model_downloaded()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            verbose=False,
        )
        self.repeat_penalty = repeat_penalty

    def _ensure_model_downloaded(self) -> str:
        """Download the GGUF file from the Hub if not already cached locally."""
        local_path = os.path.join(LOCAL_MODEL_DIR, HF_FILENAME)
        if os.path.exists(local_path):
            return local_path

        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=LOCAL_MODEL_DIR,
        )
        return downloaded_path

    def _run_model(self, text: str) -> str:
        """Run inference directly via llama-cpp-python's chat completion API."""
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            repeat_penalty=self.repeat_penalty,
            max_tokens=512,
        )
        return response["choices"][0]["message"]["content"].strip()

    def redact(self, text: str) -> RedactionResult:
        """
        Redact PII from the given text using the full two-layer pipeline
        (model + regex safety net).
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
