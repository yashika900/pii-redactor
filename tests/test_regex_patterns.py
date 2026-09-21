"""
Unit tests for the regex-based PII safety net.

Run with: pytest tests/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference.regex_patterns import apply_regex_redaction  # noqa: E402


def test_email_is_redacted():
    text = "Contact me at test@example.com for details."
    result, catches = apply_regex_redaction(text)
    assert "[EMAIL]" in result
    assert "test@example.com" not in result
    assert ("EMAIL", 1) in catches


def test_phone_is_redacted():
    text = "Call me at 415-555-2020."
    result, _ = apply_regex_redaction(text)
    assert "[PHONE]" in result
    assert "415-555-2020" not in result


def test_ssn_is_redacted():
    text = "My SSN is 123-45-6789."
    result, _ = apply_regex_redaction(text)
    assert "[SSN]" in result


def test_credit_card_is_redacted():
    text = "Card number: 4532-9876-1234-5678."
    result, _ = apply_regex_redaction(text)
    assert "[CREDIT_CARD]" in result


def test_ip_address_is_redacted():
    text = "Server ping failed at 192.168.0.44."
    result, _ = apply_regex_redaction(text)
    assert "[IP_ADDRESS]" in result


def test_credit_card_not_confused_with_phone():
    """
    Regression test: PHONE previously matched a subset of digits inside a
    credit card number before CREDIT_CARD could claim the full match.
    """
    text = "Card number 4532-9876-1234-5678 was charged."
    result, _ = apply_regex_redaction(text)
    assert "[CREDIT_CARD]" in result
    assert "[PHONE]" not in result


def test_bank_account_not_confused_with_phone():
    """
    Regression test: a long plain digit run (account number) was
    previously misclassified as a phone number.
    """
    text = "Account number 123456789012 was charged."
    result, _ = apply_regex_redaction(text)
    assert "[BANK_ACCOUNT]" in result
    assert "[PHONE]" not in result


def test_ticket_reference_not_falsely_flagged_as_license_plate():
    """
    Regression test: reference/ticket numbers (e.g. REF-9012) share a
    similar shape to license plates and were previously false-positived.
    """
    text = "Check ticket REF-9012 for status."
    result, _ = apply_regex_redaction(text)
    assert "REF-9012" in result
    assert "[LICENSE_PLATE]" not in result


def test_genuine_license_plate_with_context_is_redacted():
    text = "Her license plate ABC-1234 was reported."
    result, _ = apply_regex_redaction(text)
    assert "[LICENSE_PLATE]" in result


def test_no_pii_text_is_left_unchanged():
    text = "The store is open from 9 AM to 6 PM daily."
    result, _ = apply_regex_redaction(text)
    assert result == text


def test_dob_distinguished_from_generic_date():
    text = "DOB: 1990-05-14, appointment on 2024-08-20."
    result, catches = apply_regex_redaction(text)
    assert "[DOB]" in result
    assert "[DATE]" in result
    tags_caught = [tag for tag, _ in catches]
    assert "DOB" in tags_caught
    assert "DATE" in tags_caught
