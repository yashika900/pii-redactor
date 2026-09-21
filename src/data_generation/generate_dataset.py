"""
Synthetic PII dataset generation script.

Generates paired (original, redacted) text examples using a large teacher
model, covering diverse document types, PII categories, and difficulty
levels (messy formatting, indirect references, negative examples with no
PII). Used as the training data for fine-tuning the local redaction model.

Usage:
    export GROQ_API_KEY=your_key_here
    python -m src.data_generation.generate_dataset
"""

import json
import os
import random
import time
from typing import Dict, List, Optional

from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_TOTAL = 1200
BATCH_SIZE = 30
SLEEP_BETWEEN_CALLS = 5
OUTPUT_PATH = "data/synthetic_dataset.jsonl"
MODEL_NAME = "openai/gpt-oss-20b"

PII_TAGS_SHORT = (
    "[NAME], [EMAIL], [PHONE], [ADDRESS], [DOB], [SSN], [CREDIT_CARD], "
    "[BANK_ACCOUNT], [PASSPORT], [DRIVER_LICENSE], [IP_ADDRESS], [USERNAME], "
    "[PASSWORD], [URL], [ORG], [DATE], [LICENSE_PLATE], [MEDICAL_ID], "
    "[EMPLOYEE_ID], [AGE]"
)

# Templates span diverse document types, messiness levels, and include
# negative examples (no PII) to train precision as well as recall.
PROMPT_TEMPLATES: List[str] = [
    "Generate a realistic short email between two people that contains "
    "personal information such as a name, email address, and phone number.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a realistic informal chat/text message conversation snippet "
    "(2-4 messages) that contains personal information.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a realistic customer support ticket that contains a name, "
    "email, phone number, and an account or order ID number.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a short excerpt from a resume or job application that "
    "contains a person's name, email, phone number, and address.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a realistic medical appointment reminder or patient intake "
    "form snippet with a patient's name, date of birth, and medical "
    "record/insurance ID number.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a realistic banking or financial notification containing a "
    "name, bank account number, and partial credit card number.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a short HR/employee record excerpt containing a name, "
    "employee ID, date of birth, and address.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a travel booking confirmation or itinerary excerpt "
    "containing a name, passport number, and phone number.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a vehicle registration or parking notice excerpt containing "
    "a name, license plate number, and address.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a technical support log or bug report snippet containing a "
    "username, email, and IP address.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a social media post or forum comment containing a name and "
    "a personal social media handle/username.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a messy, informal piece of text (like a forum post or "
    "social media comment) with typos and casual grammar, that contains "
    "personal information.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a realistic short piece of text that contains personal "
    "information using unusual/obfuscated formatting (e.g., phone number "
    "spelled out in words, email written with 'at' and 'dot' spelled "
    "out).\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a realistic piece of text with personal information "
    "mentioned indirectly or contextually (e.g., 'the patient in room 4', "
    "'my manager, John, lives two blocks from me') rather than in an "
    "obvious labeled format.\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a realistic short piece of text containing personal "
    "information from a non-US context (international phone/address/ID "
    "format).\n"
    "Then provide the same text with the personal information replaced by "
    "relevant tags from: {tags}\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a realistic short piece of text that does NOT contain any "
    "personal information at all (e.g., a general announcement, weather "
    "update, or generic product description). This is a negative example.\n"
    "The redacted version should be IDENTICAL to the original since there "
    "is nothing to redact.\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',

    "Generate a realistic short piece of text that mentions a well-known "
    "public figure by name in a general news/factual context (not private "
    "info about them). This is a negative example.\n"
    "The redacted version should be IDENTICAL to the original.\n"
    'Respond ONLY in this JSON format: {{"original": "...", "redacted": "..."}}',
]


def get_prompt(template: str) -> str:
    """Fill the PII tag list into a prompt template."""
    return template.format(tags=PII_TAGS_SHORT)


def safe_generate(
    client: Groq, prompt: str, retries: int = 3
) -> Optional[Dict[str, str]]:
    """
    Call the teacher model with retry logic for rate limits and malformed
    JSON responses.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                response_format={"type": "json_object"},
                reasoning_effort="low",
            )
            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:  # noqa: BLE001 - broad by design, see branches
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str:
                wait_time = 8 * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif "Expecting value" in error_str or isinstance(e, json.JSONDecodeError):
                print("  Bad JSON, retrying...")
                time.sleep(1)
            else:
                print(f"  Unexpected error: {e}")
                return None
    return None


def generate_dataset(client: Groq) -> None:
    """Main generation loop: resumable, crash-safe, saved in batches."""
    existing_results: List[Dict[str, str]] = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            existing_results = [json.loads(line) for line in f]
        print(f"Resuming — {len(existing_results)} examples already saved")

    results = existing_results.copy()

    while len(results) < TARGET_TOTAL:
        batch_target = min(BATCH_SIZE, TARGET_TOTAL - len(results))
        print(f"\n--- Generating batch of {batch_target} "
              f"(total so far: {len(results)}) ---")

        for _ in range(batch_target):
            template = random.choice(PROMPT_TEMPLATES)
            prompt = get_prompt(template)

            data = safe_generate(client, prompt)
            if data and "original" in data and "redacted" in data:
                results.append(data)

            time.sleep(SLEEP_BETWEEN_CALLS)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Batch saved. Total so far: {len(results)}")

    print(f"\nDone. Final dataset size: {len(results)} examples "
          f"saved to {OUTPUT_PATH}")


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set the GROQ_API_KEY environment variable before running "
            "this script."
        )
    client = Groq(api_key=api_key)
    generate_dataset(client)


if __name__ == "__main__":
    main()
