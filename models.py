"""
Typed Pydantic models for the Email Triage OpenEnv environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------

class EmailCategory(str, Enum):
    URGENT = "urgent"
    NORMAL = "normal"
    SPAM = "spam"


class ActionType(str, Enum):
    CLASSIFY_EMAIL = "classify_email"
    PRIORITIZE_INBOX = "prioritize_inbox"
    DRAFT_RESPONSE = "draft_response"
    NOOP = "noop"


# ---------------------------------------------------------------------------
# Email data model
# ---------------------------------------------------------------------------

class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    category: Optional[EmailCategory] = None  # set after classification


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    goal: str
    emails: List[Email]
    inbox_state: Dict[str, Any] = Field(default_factory=dict)
    step: int = 0
    last_action_result: str = ""
    last_action_error: bool = False
    done: bool = False


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class ClassifyEmailAction(BaseModel):
    action_type: ActionType = ActionType.CLASSIFY_EMAIL
    email_id: str
    category: EmailCategory


class PrioritizeInboxAction(BaseModel):
    action_type: ActionType = ActionType.PRIORITIZE_INBOX
    ordered_ids: List[str]  # highest priority first


class DraftResponseAction(BaseModel):
    action_type: ActionType = ActionType.DRAFT_RESPONSE
    email_id: str
    response_text: str


class NoopAction(BaseModel):
    action_type: ActionType = ActionType.NOOP


# Generic action envelope accepted by /step
class Action(BaseModel):
    action_type: ActionType
    # Classification
    email_id: Optional[str] = None
    category: Optional[EmailCategory] = None
    # Prioritization
    ordered_ids: Optional[List[str]] = None
    # Response drafting
    response_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward breakdown (returned in info)
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    total: float
    components: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""