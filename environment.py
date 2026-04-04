"""
Email Triage OpenEnv Environment Core.

Implements step() / reset() / state() logic for all three tasks.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action,
    ActionType,
    Email,
    EmailCategory,
    Observation,
    StepResult,
)
from email_data import (
    CLASSIFY_EMAILS,
    PRIORITIZE_INBOX,
    CORRECT_PRIORITY_ORDER,
    RESPONSE_EMAILS,          # pool of 3 fixtures
    RESPONSE_EMAIL,           # backwards-compat alias (index 0)
    RESPONSE_CRITERIA,
)
from graders import (
    grade_classification,
    grade_classification_episode,
    grade_prioritization,
    grade_response,
)


# ---------------------------------------------------------------------------
# Task IDs
# ---------------------------------------------------------------------------

TASK_CLASSIFY = "email_classify"
TASK_PRIORITIZE = "email_prioritize"
TASK_RESPOND = "email_respond"

VALID_TASKS = [TASK_CLASSIFY, TASK_PRIORITIZE, TASK_RESPOND]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage environment.
    """

    MAX_STEPS = {
        TASK_CLASSIFY: 5,
        TASK_PRIORITIZE: 10,
        TASK_RESPOND: 15,
    }

    def __init__(self, task_id: str = TASK_CLASSIFY, email_index: int = 0):
        if task_id not in VALID_TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Valid: {VALID_TASKS}")
        self.task_id = task_id
        self.email_index = email_index % len(CLASSIFY_EMAILS)

        # For task 3: cycle through the response email pool
        self._response_email = RESPONSE_EMAILS[email_index % len(RESPONSE_EMAILS)]

        # Episode state
        self._step_count = 0
        self._done = False
        self._observation: Optional[Observation] = None

        # Task-specific tracking
        self._classifications: Dict[str, str] = {}
        self._submitted_priority: Optional[List[str]] = None
        self._submitted_response: Optional[str] = None
        self._cumulative_reward = 0.0
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset episode and return initial observation."""
        self._step_count = 0
        self._done = False
        self._classifications = {}
        self._submitted_priority = None
        self._submitted_response = None
        self._cumulative_reward = 0.0
        self._history = []

        self._observation = self._build_initial_observation()
        return copy.deepcopy(self._observation)

    def _build_initial_observation(self) -> Observation:
        if self.task_id == TASK_CLASSIFY:
            email_data = CLASSIFY_EMAILS[self.email_index]
            email = Email(
                id=email_data["id"],
                subject=email_data["subject"],
                body=email_data["body"],
                sender=email_data["sender"],
                timestamp=email_data["timestamp"],
            )
            return Observation(
                goal=(
                    f"Classify the email (id={email.id}) into one of: "
                    "urgent, normal, spam. "
                    "Use action_type='classify_email' with email_id and category."
                ),
                emails=[email],
                inbox_state={"classified": {}, "remaining": [email.id]},
                step=0,
                last_action_result="",
            )

        elif self.task_id == TASK_PRIORITIZE:
            emails = [
                Email(
                    id=e["id"],
                    subject=e["subject"],
                    body=e["body"],
                    sender=e["sender"],
                    timestamp=e["timestamp"],
                )
                for e in PRIORITIZE_INBOX
            ]
            ids = [e.id for e in emails]
            return Observation(
                goal=(
                    "Prioritize the inbox by ordering emails from most to least important. "
                    f"Email IDs: {ids}. "
                    "Use action_type='prioritize_inbox' with ordered_ids (list, highest priority first)."
                ),
                emails=emails,
                inbox_state={"ordered": [], "total": len(emails)},
                step=0,
                last_action_result="",
            )

        else:  # TASK_RESPOND
            resp = self._response_email
            email = Email(
                id=resp["id"],
                subject=resp["subject"],
                body=resp["body"],
                sender=resp["sender"],
                timestamp=resp["timestamp"],
            )
            criteria_list = list(RESPONSE_CRITERIA.keys())
            return Observation(
                goal=(
                    f"Draft a professional response to email id={email.id} "
                    f"from {email.sender} about: '{email.subject}'. "
                    "Use action_type='draft_response' with email_id and response_text. "
                    f"Your response must include: {', '.join(criteria_list)}."
                ),
                emails=[email],
                inbox_state={"response_drafted": False, "attempts": 0},
                step=0,
                last_action_result="",
            )

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """Execute an action and return the next observation + reward."""
        if self._done:
            return StepResult(
                observation=copy.deepcopy(self._observation),
                reward=0.0,
                done=True,
                info={"warning": "Episode already done. Call reset()."},
            )

        self._step_count += 1
        reward = 0.0
        info: Dict[str, Any] = {}
        error = False

        # ---- Dispatch by task -------------------------------------------
        if self.task_id == TASK_CLASSIFY:
            reward, info, error = self._step_classify(action)

        elif self.task_id == TASK_PRIORITIZE:
            reward, info, error = self._step_prioritize(action)

        else:
            reward, info, error = self._step_respond(action)

        # ---- Check episode end ------------------------------------------
        max_steps = self.MAX_STEPS[self.task_id]
        if self._step_count >= max_steps:
            self._done = True

        self._cumulative_reward = round(self._cumulative_reward + reward, 4)
        self._history.append(
            {
                "step": self._step_count,
                "action_type": action.action_type,
                "reward": reward,
                "done": self._done,
            }
        )

        # Update observation
        self._observation.step = self._step_count
        self._observation.done = self._done
        self._observation.last_action_error = error

        return StepResult(
            observation=copy.deepcopy(self._observation),
            reward=reward,
            done=self._done,
            info=info,
        )

    # ------------------------------------------------------------------
    # Task-specific step handlers
    # ------------------------------------------------------------------

    def _step_classify(
        self, action: Action
    ) -> Tuple[float, Dict[str, Any], bool]:
        if action.action_type == ActionType.NOOP:
            self._observation.last_action_result = "noop — no action taken."
            return 0.0, {"action": "noop"}, False

        if action.action_type != ActionType.CLASSIFY_EMAIL:
            msg = f"Invalid action for classify task: {action.action_type}"
            self._observation.last_action_result = msg
            return 0.0, {"error": msg}, True

        if not action.email_id or not action.category:
            msg = "classify_email requires email_id and category."
            self._observation.last_action_result = msg
            return 0.0, {"error": msg}, True

        reward, info = grade_classification(action.email_id, action.category)
        self._classifications[action.email_id] = action.category

        # Mark email as classified in state
        remaining = self._observation.inbox_state.get("remaining", [])
        if action.email_id in remaining:
            remaining.remove(action.email_id)
        self._observation.inbox_state["classified"][action.email_id] = action.category
        self._observation.inbox_state["remaining"] = remaining

        result_str = (
            f"Classified {action.email_id} as '{action.category}' — "
            f"{'CORRECT' if reward == 1.0 else 'WRONG'}."
        )
        self._observation.last_action_result = result_str

        # Episode ends as soon as all emails in the observation are classified
        if not remaining:
            self._done = True

        return reward, info, False

    def _step_prioritize(
            self, action: Action
        ) -> Tuple[float, Dict[str, Any], bool]:
            if action.action_type == ActionType.NOOP:
                self._observation.last_action_result = "noop — no action taken."
                return 0.0, {"action": "noop"}, False

            if action.action_type != ActionType.PRIORITIZE_INBOX:
                msg = f"Invalid action for prioritize task: {action.action_type}"
                self._observation.last_action_result = msg
                return 0.0, {"error": msg}, True

            if not action.ordered_ids:
                msg = "prioritize_inbox requires ordered_ids list."
                self._observation.last_action_result = msg
                return 0.0, {}, True

            reward, info = grade_prioritization(action.ordered_ids)
            self._submitted_priority = action.ordered_ids
            self._observation.inbox_state["ordered"] = action.ordered_ids

            attempts = self._observation.inbox_state.get("attempts", 0) + 1
            self._observation.inbox_state["attempts"] = attempts
            self._observation.inbox_state["last_reward"] = reward

            self._observation.last_action_result = (
                f"Submitted priority order (attempt #{attempts}) — reward: {reward:.3f}. "
                f"Correct order: {CORRECT_PRIORITY_ORDER}. "
                f"Refine your ordering and try again, or a perfect score ends the episode."
            )

            # End on perfect score, otherwise let the agent try again within max_steps
            if reward >= 1.0:
                self._done = True

            return reward, info, False

    def _step_respond(
        self, action: Action
    ) -> Tuple[float, Dict[str, Any], bool]:
        if action.action_type == ActionType.NOOP:
            self._observation.last_action_result = "noop — no action taken."
            return 0.0, {"action": "noop"}, False

        if action.action_type != ActionType.DRAFT_RESPONSE:
            msg = f"Invalid action for respond task: {action.action_type}"
            self._observation.last_action_result = msg
            return 0.0, {"error": msg}, True

        if not action.email_id or not action.response_text:
            msg = "draft_response requires email_id and response_text."
            self._observation.last_action_result = msg
            return 0.0, {"error": msg}, True

        reward, info = grade_response(action.email_id, action.response_text)
        self._submitted_response = action.response_text

        attempts = self._observation.inbox_state.get("attempts", 0) + 1
        self._observation.inbox_state["attempts"] = attempts
        self._observation.inbox_state["response_drafted"] = True
        self._observation.inbox_state["last_reward"] = reward

        # Allow multiple attempts; episode ends on perfect score or max steps
        if reward >= 1.0:
            self._done = True

        self._observation.last_action_result = (
            f"Response graded: {reward:.3f}/1.0 on attempt #{attempts}. "
            f"Criteria met: {[k for k,v in info.get('criteria',{}).items() if isinstance(v,dict) and v.get('met')]}."
        )
        return reward, info, False

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        obs = self._observation
        return {
                "task_id": self.task_id,
                "step": self._step_count,
                "done": self._done,
                "cumulative_reward": self._cumulative_reward,
                "best_step_reward": round(max(
                    (h["reward"] for h in self._history), default=0.0
                ), 4),
                "observation": obs.model_dump() if obs else None,
                "history": self._history,
                "task_specific": {
                    "classifications": self._classifications,
                    "submitted_priority": self._submitted_priority,
                    "response_preview": (
                        (self._submitted_response or "")[:200]
                        if self._submitted_response
                        else None
                    ),
                },
            }