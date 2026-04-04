"""
Unit tests for graders and environment.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from graders import (
    grade_classification,
    grade_classification_episode,
    grade_prioritization,
    grade_response,
)
from environment import EmailTriageEnv, TASK_CLASSIFY, TASK_PRIORITIZE, TASK_RESPOND
from models import Action, ActionType, EmailCategory
from email_data import CORRECT_PRIORITY_ORDER


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestClassificationGrader:
    def test_correct_urgent(self):
        reward, info = grade_classification("e001", "urgent")
        assert reward == 1.0
        assert info["result"] == "correct"

    def test_correct_spam(self):
        reward, info = grade_classification("e002", "spam")
        assert reward == 1.0

    def test_correct_normal(self):
        reward, info = grade_classification("e003", "normal")
        assert reward == 1.0

    def test_wrong_category(self):
        reward, info = grade_classification("e001", "spam")  # is urgent
        assert reward == 0.0
        assert info["result"] == "wrong"

    def test_invalid_category(self):
        reward, info = grade_classification("e001", "junk")
        assert reward == 0.0
        assert "error" in info

    def test_unknown_email_id(self):
        reward, info = grade_classification("e999", "urgent")
        assert reward == 0.0
        assert "error" in info

    def test_case_insensitive(self):
        reward, info = grade_classification("e001", "URGENT")
        assert reward == 1.0

    def test_episode_full_correct(self):
        correct = {
            "e001": "urgent",
            "e002": "spam",
            "e003": "normal",
        }
        reward, info = grade_classification_episode(correct)
        assert reward == 1.0
        assert info["accuracy"] == 1.0

    def test_episode_partial_credit(self):
        mixed = {
            "e001": "urgent",  # correct
            "e002": "normal",  # wrong (spam)
        }
        reward, info = grade_classification_episode(mixed)
        assert reward == 0.5


class TestPrioritizationGrader:
    def test_perfect_order(self):
        reward, info = grade_prioritization(CORRECT_PRIORITY_ORDER)
        assert reward >= 0.9  # should be near perfect

    def test_reversed_order(self):
        reversed_order = list(reversed(CORRECT_PRIORITY_ORDER))
        reward, info = grade_prioritization(reversed_order)
        assert reward < 0.5  # bad ordering

    def test_missing_ids(self):
        reward, info = grade_prioritization(["p001", "p002"])
        assert reward <= 0.2
        assert "error" in info or "missing" in info

    def test_urgents_at_top(self):
        order = ["p001", "p003", "p002", "p005", "p004"]
        reward, info = grade_prioritization(order)
        assert info["top2_urgent_correct"] is True

    def test_spam_at_bottom(self):
        order = ["p001", "p003", "p002", "p005", "p004"]
        reward, info = grade_prioritization(order)
        assert info["spam_at_bottom"] is True

    def test_reward_in_range(self):
        for order in [
            CORRECT_PRIORITY_ORDER,
            list(reversed(CORRECT_PRIORITY_ORDER)),
        ]:
            reward, _ = grade_prioritization(order)
            assert 0.0 <= reward <= 1.0


class TestResponseGrader:
    # --- r001: duplicate billing charge ---
    GOOD_RESPONSE_R001 = """
Dear Jane,

Thank you for reaching out. We sincerely apologize for the inconvenience caused
by the duplicate charge on your account. We understand how frustrating this must be,
especially as a loyal customer of two years.

We have reviewed your account (jane.doe@example.com) and can confirm that you were
indeed charged twice — once on March 1st and again on March 8th. We will immediately
process a full refund of $49.99 to your original payment method within 3–5 business days.

Could you please confirm your billing account number or the last four digits of your
card so we can verify the transaction on our end?

We value your continued trust and are committed to resolving this promptly.

Kind regards,
Customer Support Team
"""

    # --- r002: wrong item shipped ---
    GOOD_RESPONSE_R002 = """
Dear Tom,

We sincerely apologize for the inconvenience caused by receiving the wrong item in
your order #78423. We understand how frustrating this must be, especially with a
work deadline approaching.

We will immediately arrange a replacement shipment of the correct blue 16GB USB drive
and send a prepaid return label for the red mouse you received in error.

Could you please confirm your order number and account email so we can verify the
details and prioritize your replacement?

We appreciate your patience and value your business.

Kind regards,
Customer Support Team
"""

    # --- r003: account locked ---
    GOOD_RESPONSE_R003 = """
Dear Sarah,

We sincerely apologize for the inconvenience of your account being locked and the
disruption this has caused to your work.

We will immediately investigate and unlock your account as soon as possible. Our
team takes account access issues seriously and will work to restore your access
without delay.

Could you please confirm your account email address so we can verify your identity
and resolve this issue promptly?

We appreciate your patience and understand the urgency given your presentation tomorrow.

Kind regards,
Customer Support Team
"""

    def test_good_response_r001_high_score(self):
        reward, info = grade_response("r001", self.GOOD_RESPONSE_R001)
        assert reward >= 0.8, f"Expected >= 0.8, got {reward}. Info: {info}"

    def test_good_response_r002_high_score(self):
        reward, info = grade_response("r002", self.GOOD_RESPONSE_R002)
        assert reward >= 0.8, f"Expected >= 0.8, got {reward}. Info: {info}"

    def test_good_response_r003_high_score(self):
        reward, info = grade_response("r003", self.GOOD_RESPONSE_R003)
        assert reward >= 0.8, f"Expected >= 0.8, got {reward}. Info: {info}"

    def test_empty_response(self):
        reward, info = grade_response("r001", "")
        assert reward == 0.0

    def test_short_response(self):
        reward, info = grade_response("r001", "Sorry about that.")
        assert reward <= 0.3

    def test_wrong_email_id(self):
        reward, info = grade_response("r999", "Any response text here.")
        assert reward == 0.0
        assert "error" in info

    def test_reward_in_range_r001(self):
        reward, _ = grade_response("r001", self.GOOD_RESPONSE_R001)
        assert 0.0 <= reward <= 1.0

    def test_reward_in_range_r002(self):
        reward, _ = grade_response("r002", self.GOOD_RESPONSE_R002)
        assert 0.0 <= reward <= 1.0

    def test_reward_in_range_r003(self):
        reward, _ = grade_response("r003", self.GOOD_RESPONSE_R003)
        assert 0.0 <= reward <= 1.0

    def test_missing_criteria_reduces_score(self):
        # No resolution mention
        partial = "Dear Jane, I apologize for any issues. Please contact us. Regards, Support"
        reward, _ = grade_response("r001", partial)
        full_reward, _ = grade_response("r001", self.GOOD_RESPONSE_R001)
        assert reward < full_reward

    def test_r002_wrong_item_acknowledged(self):
        reward, info = grade_response("r002", self.GOOD_RESPONSE_R002)
        assert info["criteria"]["acknowledges_issue"]["met"] is True

    def test_r003_account_lock_acknowledged(self):
        reward, info = grade_response("r003", self.GOOD_RESPONSE_R003)
        assert info["criteria"]["acknowledges_issue"]["met"] is True

    def test_r002_replacement_resolution(self):
        reward, info = grade_response("r002", self.GOOD_RESPONSE_R002)
        assert info["criteria"]["mentions_resolution"]["met"] is True

    def test_r003_unlock_resolution(self):
        reward, info = grade_response("r003", self.GOOD_RESPONSE_R003)
        assert info["criteria"]["mentions_resolution"]["met"] is True


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestClassifyEnvironment:
    def test_reset_returns_observation(self):
        env = EmailTriageEnv(TASK_CLASSIFY, email_index=0)
        obs = env.reset()
        assert obs.goal
        assert len(obs.emails) == 1
        assert obs.step == 0
        assert not obs.done

    def test_correct_classification_reward_1(self):
        env = EmailTriageEnv(TASK_CLASSIFY, email_index=0)
        obs = env.reset()
        email_id = obs.emails[0].id
        # e001 is urgent
        action = Action(
            action_type=ActionType.CLASSIFY_EMAIL,
            email_id=email_id,
            category=EmailCategory.URGENT,
        )
        result = env.step(action)
        assert result.reward == 1.0
        assert result.done  # Single email → done after classification

    def test_wrong_classification_reward_0(self):
        env = EmailTriageEnv(TASK_CLASSIFY, email_index=0)
        env.reset()
        action = Action(
            action_type=ActionType.CLASSIFY_EMAIL,
            email_id="e001",
            category=EmailCategory.SPAM,
        )
        result = env.step(action)
        assert result.reward == 0.0

    def test_noop_action(self):
        env = EmailTriageEnv(TASK_CLASSIFY, email_index=0)
        env.reset()
        action = Action(action_type=ActionType.NOOP)
        result = env.step(action)
        assert result.reward == 0.0
        assert not result.done

    def test_state_method(self):
        env = EmailTriageEnv(TASK_CLASSIFY)
        env.reset()
        state = env.state()
        assert "task_id" in state
        assert state["task_id"] == TASK_CLASSIFY
        assert "step" in state


class TestPrioritizeEnvironment:
    def test_reset(self):
        env = EmailTriageEnv(TASK_PRIORITIZE)
        obs = env.reset()
        assert len(obs.emails) == 5
        assert not obs.done

    def test_correct_priority_high_reward(self):
        env = EmailTriageEnv(TASK_PRIORITIZE)
        env.reset()
        action = Action(
            action_type=ActionType.PRIORITIZE_INBOX,
            ordered_ids=CORRECT_PRIORITY_ORDER,
        )
        result = env.step(action)
        assert result.reward >= 0.9
        assert result.done

    def test_missing_ordered_ids_error(self):
        env = EmailTriageEnv(TASK_PRIORITIZE)
        env.reset()
        action = Action(action_type=ActionType.PRIORITIZE_INBOX)
        result = env.step(action)
        assert result.reward == 0.0
        assert result.observation.last_action_error


class TestRespondEnvironment:
    def test_reset(self):
        env = EmailTriageEnv(TASK_RESPOND)
        obs = env.reset()
        assert len(obs.emails) == 1
        assert not obs.done

    def test_reset_cycles_email_pool(self):
        """email_index cycles through r001, r002, r003."""
        for idx, expected_id in enumerate(["r001", "r002", "r003"]):
            env = EmailTriageEnv(TASK_RESPOND, email_index=idx)
            obs = env.reset()
            assert obs.emails[0].id == expected_id

    def test_good_response_r001(self):
        env = EmailTriageEnv(TASK_RESPOND, email_index=0)
        env.reset()
        good_text = (
            "Dear Jane, we sincerely apologize for the duplicate charge on your account. "
            "We can confirm you were double-charged and will process a full refund "
            "of $49.99 immediately. Please verify your account details so we can "
            "complete the refund process. Thank you for your patience. Kind regards, Support."
        )
        action = Action(
            action_type=ActionType.DRAFT_RESPONSE,
            email_id="r001",
            response_text=good_text,
        )
        result = env.step(action)
        assert result.reward > 0.5

    def test_good_response_r002(self):
        env = EmailTriageEnv(TASK_RESPOND, email_index=1)
        env.reset()
        good_text = (
            "Dear Tom, we sincerely apologize for shipping the wrong item in your order. "
            "We will immediately arrange a replacement of the correct item and send a "
            "prepaid return label. Please confirm your order number so we can verify "
            "the details and prioritize your replacement. We appreciate your patience. "
            "Kind regards, Support."
        )
        action = Action(
            action_type=ActionType.DRAFT_RESPONSE,
            email_id="r002",
            response_text=good_text,
        )
        result = env.step(action)
        assert result.reward > 0.5

    def test_good_response_r003(self):
        env = EmailTriageEnv(TASK_RESPOND, email_index=2)
        env.reset()
        good_text = (
            "Dear Sarah, we sincerely apologize for the inconvenience of your account "
            "being locked. We will immediately investigate and unlock your account. "
            "Please confirm your account email so we can verify your identity and "
            "resolve this issue promptly. We appreciate your patience. Kind regards, Support."
        )
        action = Action(
            action_type=ActionType.DRAFT_RESPONSE,
            email_id="r003",
            response_text=good_text,
        )
        result = env.step(action)
        assert result.reward > 0.5

    def test_empty_response_error(self):
        env = EmailTriageEnv(TASK_RESPOND)
        env.reset()
        action = Action(
            action_type=ActionType.DRAFT_RESPONSE,
            email_id="r001",
            response_text="",
        )
        result = env.step(action)
        assert result.reward == 0.0

    def test_done_after_max_steps(self):
        env = EmailTriageEnv(TASK_RESPOND)
        env.reset()
        action = Action(action_type=ActionType.NOOP)
        result = None
        for _ in range(env.MAX_STEPS[TASK_RESPOND]):
            result = env.step(action)
        assert result.done

    def test_reward_in_range(self):
        env = EmailTriageEnv(TASK_RESPOND)
        env.reset()
        action = Action(
            action_type=ActionType.DRAFT_RESPONSE,
            email_id="r001",
            response_text="Some response text here that is long enough to pass minimum length.",
        )
        result = env.step(action)
        assert 0.0 <= result.reward <= 1.0