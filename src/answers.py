"""Strict numeric answer extraction and scoring; no API or dataset access."""

import re
from decimal import Decimal


_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_VALUE = re.compile(rf"\s*\$?\s*({_NUMBER})\s*(?:dollars?|cents?)?\s*[.!]?\s*", re.I)
_FINAL = re.compile(r"(?:\bAnswer\s*:|####)\s*([^\r\n]*)", re.I)


def normalize_number(text: str) -> str:
    """Canonical decimal string, or empty for invalid/ambiguous input.

    Fractions, equations, percentages, and malformed comma groups are rejected.
    No float rounding is used: 3, 3.0 and 3.00 compare equal.
    """
    match = _VALUE.fullmatch(text.strip().replace("−", "-"))
    if not match:
        return ""
    value = Decimal(match.group(1).replace(",", ""))
    if value == 0:
        return "0"
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def extract_gsm8k_answer(response_text: str) -> str:
    """Use the last final-answer marker, or a response containing only a number.

    Never guess from numbers in unfinished reasoning. An invalid final marker
    stays invalid even if an earlier marker contained a valid number.
    """
    matches = list(_FINAL.finditer(response_text))
    candidate = matches[-1].group(1) if matches else response_text
    return normalize_number(candidate)


def extract_gold_answer(raw_answer: str) -> str:
    """Require GSM8K's answer-key marker; invalid keys stop evaluation."""
    if "####" not in raw_answer:
        raise ValueError("GSM8K answer key is missing the #### marker")
    answer = normalize_number(raw_answer.rsplit("####", 1)[1])
    if not answer:
        raise ValueError("GSM8K answer key has an invalid final number")
    return answer


def is_correct(prediction: str, gold_answer: str) -> bool:
    gold = normalize_number(gold_answer)
    if not gold:
        raise ValueError("Cannot score against an invalid gold answer")
    predicted = normalize_number(prediction)
    return bool(predicted) and predicted == gold
