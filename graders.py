"""
Deterministic graders for all three Email Triage tasks.
Each grader returns a float strictly in (0.0, 1.0) — exclusive bounds
required by the OpenEnv validator.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from models import EmailCategory
from email_data import (
    CLASSIFY_EMAILS,
    CORRECT_PRIORITY_ORDER,
    RESPONSE_CRITERIA,
    RESPONSE_EMAILS,
)

# Build a lookup so we can retrieve any response email by id
_RESPONSE_EMAIL_BY_ID: Dict[str, Dict] = {e["id"]: e for e in RESPONSE_EMAILS}

# ---------------------------------------------------------------------------
# Score clamping — validator requires strictly (0, 1), not 0.0 or 1.0
# ---------------------------------------------------------------------------

_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) as required by the validator."""
    return max(_SCORE_MIN, min(_SCORE_MAX, round(score, 4)))


# ---------------------------------------------------------------------------
# Task 1: Email Classification Grader
# ---------------------------------------------------------------------------

def grade_classification(
    email_id: str,
    predicted_category: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Grades a single email classification.
    Returns (reward, info_dict). Reward strictly in (0, 1).
    """
    label_map = {e["id"]: e["label"] for e in CLASSIFY_EMAILS}
    correct_label = label_map.get(email_id)

    if correct_label is None:
        return _SCORE_MIN, {"error": f"Unknown email_id: {email_id}"}

    try:
        predicted = EmailCategory(predicted_category.lower())
    except ValueError:
        return _SCORE_MIN, {
            "error": f"Invalid category: {predicted_category}",
            "valid_categories": [c.value for c in EmailCategory],
        }

    correct = predicted == correct_label
    # 0.99 for correct, 0.01 for wrong — strictly within (0, 1)
    reward = _SCORE_MAX if correct else _SCORE_MIN

    return reward, {
        "email_id": email_id,
        "predicted": predicted.value,
        "correct": correct_label.value,
        "result": "correct" if correct else "wrong",
    }


def grade_classification_episode(
    classifications: Dict[str, str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Grade an entire classification episode. Partial credit: fraction correct.
    """
    if not classifications:
        return _SCORE_MIN, {"error": "No classifications provided"}

    total = len(classifications)
    correct_count = 0
    details = {}

    for email_id, predicted in classifications.items():
        reward, info = grade_classification(email_id, predicted)
        if reward >= 0.5:
            correct_count += 1
        details[email_id] = info

    raw = correct_count / total
    reward = _clamp(raw)
    return reward, {
        "correct": correct_count,
        "total": total,
        "accuracy": raw,
        "reward": reward,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Task 2: Email Prioritization Grader
# ---------------------------------------------------------------------------

def _kendall_tau_distance(list_a: List[str], list_b: List[str]) -> float:
    """Normalised Kendall tau distance. Returns 0.0 (identical) to 1.0 (reversed)."""
    pos_a = {v: i for i, v in enumerate(list_a)}
    pos_b = {v: i for i, v in enumerate(list_b)}

    inversions = 0
    comparisons = 0
    items = [x for x in list_a if x in pos_b]

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            comparisons += 1
            a_order = pos_a[items[i]] < pos_a[items[j]]
            b_order = pos_b[items[i]] < pos_b[items[j]]
            if a_order != b_order:
                inversions += 1

    if comparisons == 0:
        return 0.0
    return inversions / comparisons


def grade_prioritization(
    submitted_order: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Grade the inbox prioritization task using Kendall tau similarity.
    Reward strictly in (0, 1).
    """
    correct = CORRECT_PRIORITY_ORDER
    expected_set = set(correct)
    submitted_set = set(submitted_order)

    missing = expected_set - submitted_set
    extra = submitted_set - expected_set

    if missing:
        return _SCORE_MIN, {
            "error": "Missing email IDs in submission",
            "missing": list(missing),
            "extra": list(extra),
        }

    filtered = [x for x in submitted_order if x in expected_set]

    tau_dist = _kendall_tau_distance(correct, filtered)
    tau_similarity = 1.0 - tau_dist

    urgent_ids = {"p001", "p003"}
    top2_bonus = 0.1 if set(submitted_order[:2]) == urgent_ids else 0.0

    spam_ids = {"p004"}
    spam_bottom_bonus = 0.05 if submitted_order[-1] in spam_ids else 0.0

    raw = tau_similarity * 0.85 + top2_bonus + spam_bottom_bonus
    reward = _clamp(raw)

    return reward, {
        "tau_similarity": round(tau_similarity, 4),
        "top2_urgent_correct": set(submitted_order[:2]) == urgent_ids,
        "spam_at_bottom": submitted_order[-1] in spam_ids,
        "submitted_order": submitted_order,
        "correct_order": correct,
        "reward": reward,
    }


# ---------------------------------------------------------------------------
# Task 3: Email Response Drafting Grader
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = [
    r"\b(dear|hello|hi|good\s+(morning|afternoon|evening))\b",
]
_APOLOGY_PATTERNS = [
    r"\b(sorry|apologise|apologize|apologies|regret|inconvenience)\b",
]
_VERIFY_PATTERNS = [
    r"\b(please\s+(provide|share|confirm|send|verify).{0,30}(account|order|transaction|reference))\b",
    r"\b(can\s+you\s+(provide|share|confirm|send).{0,30}(account|order|transaction|reference))\b",
    r"\b(could\s+you\s+(provide|share|confirm|send).{0,30}(account|order|transaction|reference))\b",
    r"\b(we\s+(will\s+)?(need|require).{0,30}(account|order|transaction|reference)\s+(number|detail|id|info))\b",
    r"\b(verify\s+(your\s+)?(account|order|transaction|identity|details))\b",
    r"\b(confirm\s+(your\s+)?(account|order|transaction|details|identity))\b",
]
_PROFESSIONAL_PATTERNS = [
    r"\b(we\s+(sincerely\s+)?understand\s+(your|this|how))\b",
    r"\b(we\s+value\s+your\s+(loyalty|business|patronage|trust|patience))\b",
    r"\b(we\s+(sincerely\s+)?apologize\s+for\s+the)\b",
    r"\b(we\s+(sincerely\s+)?apologise\s+for\s+the)\b",
    r"\b(we\s+take\s+(this|billing|these|your)\s+(matter|error|issue|concern|request)s?\s+seriously)\b",
    r"\b(your\s+satisfaction\s+is\s+(our|a)\s+(top\s+)?priority)\b",
    r"\b(we\s+appreciate\s+your\s+(patience|understanding|loyalty|business))\b",
    r"\b(as\s+a\s+(valued|loyal)\s+customer)\b",
]
_SIGNOFF_PATTERNS = [
    r"\b(sincerely|regards|best regards|kind regards|warm regards|yours)\b",
]

_ACKNOWLEDGE_PATTERNS_BY_ID: Dict[str, List[str]] = {
    "r001": [r"\b(double.?charge|charged twice|duplicate.?charge|billing error|two charges|charged again)\b"],
    "r002": [
        r"\b(wrong item|incorrect item|wrong product|incorrect order|wrong.{0,15}shipped|wrong.{0,15}received)\b",
        r"\b(received.{0,20}wrong|sent.{0,20}wrong|shipped.{0,20}wrong)\b",
        r"\b(order.{0,20}incorrect|incorrect.{0,20}order)\b",
    ],
    "r003": [
        r"\b(account.{0,20}lock(ed)?|lock(ed)?.{0,20}account|account.{0,20}access|unable to access)\b",
        r"\b(cannot access|can.t access|access.{0,20}issue|account.{0,20}suspend)\b",
    ],
}

_RESOLUTION_PATTERNS_BY_ID: Dict[str, List[str]] = {
    "r001": [r"\b(refund|reimburse|credit|reverse|return.{0,20}charge|process.{0,20}refund)\b"],
    "r002": [
        r"\b(replacement|replace|correct item|right item|resend|re.?send|return label|prepaid label)\b",
        r"\b(send.{0,20}correct|dispatch.{0,20}correct|ship.{0,20}correct|arrange.{0,20}replace)\b",
    ],
    "r003": [
        r"\b(unlock|restore.{0,20}access|re.?activate|reinstate|resolve.{0,20}account)\b",
        r"\b(account.{0,20}unlock|access.{0,20}restor|investigat.{0,20}(issue|lock|account))\b",
    ],
}


def _build_criteria_patterns(email_id: str) -> Dict[str, List[str]]:
    return {
        "has_greeting": _GREETING_PATTERNS,
        "acknowledges_issue": _ACKNOWLEDGE_PATTERNS_BY_ID.get(email_id, _ACKNOWLEDGE_PATTERNS_BY_ID["r001"]),
        "apologizes": _APOLOGY_PATTERNS,
        "mentions_resolution": _RESOLUTION_PATTERNS_BY_ID.get(email_id, _RESOLUTION_PATTERNS_BY_ID["r001"]),
        "requests_verification": _VERIFY_PATTERNS,
        "professional_tone": _PROFESSIONAL_PATTERNS,
        "has_signoff": _SIGNOFF_PATTERNS,
    }


def _check_criterion(text: str, patterns: List[str]) -> bool:
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def grade_response(
    email_id: str,
    response_text: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Grade a drafted response. Reward strictly in (0, 1).
    """
    if email_id not in _RESPONSE_EMAIL_BY_ID:
        return _SCORE_MIN, {
            "error": f"Unknown email_id for response task: {email_id}",
            "valid_ids": list(_RESPONSE_EMAIL_BY_ID.keys()),
        }

    if not response_text or len(response_text.strip()) < 20:
        return _SCORE_MIN, {"error": "Response too short or empty"}

    criteria_patterns = _build_criteria_patterns(email_id)
    total_reward = 0.0
    criteria_results = {}

    for criterion, meta in RESPONSE_CRITERIA.items():
        patterns = criteria_patterns.get(criterion, [])
        met = _check_criterion(response_text, patterns) if patterns else False
        weight = meta["weight"]
        earned = weight if met else 0.0
        total_reward += earned
        criteria_results[criterion] = {
            "met": met,
            "weight": weight,
            "earned": earned,
            "description": meta["description"],
        }

    # Length penalty for very short responses
    word_count = len(response_text.split())
    if word_count < 50:
        length_penalty = (50 - word_count) / 50 * 0.2
        total_reward = max(0.0, total_reward - length_penalty)
        criteria_results["length_penalty"] = {
            "word_count": word_count,
            "penalty": round(length_penalty, 4),
        }

    reward = _clamp(total_reward)

    return reward, {
        "email_id": email_id,
        "reward": reward,
        "word_count": word_count,
        "criteria": criteria_results,
    }