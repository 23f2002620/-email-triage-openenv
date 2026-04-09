"""
Baseline inference script for the Email Triage OpenEnv environment.

Required environment variables:
    API_BASE_URL   - LLM API endpoint (OpenAI-compatible)
    MODEL_NAME     - Model identifier to use for inference
    HF_TOKEN       - Your Hugging Face / API key
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — defaults set only for API_BASE_URL and MODEL_NAME
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "<your-api-base-url>")
MODEL_NAME: str = os.getenv("MODEL_NAME", "<your-action-model-name>")
HF_TOKEN: str = os.getenv("HF_TOKEN")

LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME")

ENV_MODE: str = os.getenv("ENV_MODE", "local").lower()
SPACE_URL: str = (os.getenv("SPACE_URL") or "").rstrip("/")

MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 512

# All LLM calls use the OpenAI client configured via these variables
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")



# ---------------------------------------------------------------------------
# Score clamping — validator requires strictly (0, 1), never 0.0 or 1.0
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) as required by the OpenEnv validator."""
    return max(0.01, min(0.99, float(score)))

# ---------------------------------------------------------------------------
# Local environment import
# ---------------------------------------------------------------------------

if ENV_MODE == "local":
    from environment import EmailTriageEnv, TASK_CLASSIFY, TASK_PRIORITIZE, TASK_RESPOND
    from models import Action, ActionType, EmailCategory
    from email_data import CLASSIFY_EMAILS, RESPONSE_EMAILS

    TASKS = (
        [{"task_id": TASK_CLASSIFY, "email_index": i} for i in range(len(CLASSIFY_EMAILS))]
        + [{"task_id": TASK_PRIORITIZE, "email_index": 0}]
        + [{"task_id": TASK_RESPOND, "email_index": i} for i in range(len(RESPONSE_EMAILS))]
    )
else:
    TASK_CLASSIFY = "email_classify"
    TASK_PRIORITIZE = "email_prioritize"
    TASK_RESPOND = "email_respond"
    TASKS = (
        [{"task_id": TASK_CLASSIFY, "email_index": i} for i in range(10)]
        + [{"task_id": TASK_PRIORITIZE, "email_index": 0}]
        + [{"task_id": TASK_RESPOND, "email_index": i} for i in range(3)]
    )

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_reset(task_id: str, email_index: int) -> Tuple[str, Dict[str, Any]]:
    import requests
    resp = requests.post(
        f"{SPACE_URL}/reset",
        json={"task_id": task_id, "email_index": email_index},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["session_id"], data["observation"]


def _http_step(session_id: str, action_dict: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    resp = requests.post(
        f"{SPACE_URL}/step",
        json={"session_id": session_id, **action_dict},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI email triage assistant operating inside an OpenEnv environment.
    Respond with a single valid JSON action object using exactly one of these schemas:

    1. {"action_type": "classify_email", "email_id": "<id>", "category": "urgent"|"normal"|"spam"}
    2. {"action_type": "prioritize_inbox", "ordered_ids": ["<id1>", "<id2>", ...]}
    3. {"action_type": "draft_response", "email_id": "<id>", "response_text": "<full reply>"}
    4. {"action_type": "noop"}

    Output ONLY valid JSON. No prose, no markdown fences, no explanation.
    For draft_response: include greeting, acknowledgment, apology, resolution, sign-off.
""").strip()


def build_user_prompt(observation: Dict[str, Any]) -> str:
    emails_summary = []
    for email in observation.get("emails", []):
        emails_summary.append(
            f"  ID: {email['id']}\n"
            f"  Subject: {email['subject']}\n"
            f"  From: {email['sender']}\n"
            f"  Body: {email['body'][:400]}"
        )
    emails_text = "\n\n".join(emails_summary) if emails_summary else "  (no emails)"
    return (
        f"GOAL: {observation.get('goal', '(none)')}\n"
        f"STEP: {observation.get('step', 0)}\n\n"
        f"EMAILS IN INBOX:\n{emails_text}\n\n"
        f"INBOX STATE: {json.dumps(observation.get('inbox_state', {}))}\n"
        f"LAST ACTION RESULT: {observation.get('last_action_result') or '(none)'}\n"
        f"LAST ACTION ERROR: {observation.get('last_action_error', False)}\n\n"
        f"Respond with a single valid JSON action."
    )


def parse_action(response_text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"^```(?:json)?\s*", "", response_text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None

# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_episode_local(task_id: str, email_index: int = 0) -> Tuple[float, int]:
    env = EmailTriageEnv(task_id=task_id, email_index=email_index)
    obs_dict = env.reset().model_dump()
    final_reward = 0.0
    steps_taken = 0

    for step_num in range(1, MAX_STEPS + 1):
        if obs_dict.get("done", False):
            break

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(obs_dict)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception:
            response_text = '{"action_type": "noop"}'

        action_dict = parse_action(response_text) or {"action_type": "noop"}

        try:
            action = Action(
                action_type=ActionType(action_dict.get("action_type", "noop")),
                email_id=action_dict.get("email_id"),
                category=EmailCategory(action_dict["category"].lower())
                    if action_dict.get("category") else None,
                ordered_ids=action_dict.get("ordered_ids"),
                response_text=action_dict.get("response_text"),
            )
        except Exception:
            action = Action(action_type=ActionType.NOOP)

        result = env.step(action)
        reward = _clamp(result.reward)
        steps_taken = step_num
        final_reward = max(final_reward, reward)

        # Required structured output — flush=True so validator captures immediately
        print(f"[STEP] step={step_num} reward={reward:.4f}", flush=True)

        obs_dict = result.observation.model_dump()
        if result.done:
            break

    return final_reward, steps_taken


def run_episode_http(task_id: str, email_index: int = 0) -> Tuple[float, int]:
    session_id, obs_dict = _http_reset(task_id, email_index)
    final_reward = 0.0
    steps_taken = 0

    for step_num in range(1, MAX_STEPS + 1):
        if obs_dict.get("done", False):
            break

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(obs_dict)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception:
            response_text = '{"action_type": "noop"}'

        action_dict = parse_action(response_text) or {"action_type": "noop"}
        result = _http_step(session_id, action_dict)
        obs_dict = result["observation"]
        reward = _clamp(result["reward"])
        steps_taken = step_num
        final_reward = max(final_reward, reward)

        # Required structured output — flush=True so validator captures immediately
        print(f"[STEP] step={step_num} reward={reward:.4f}", flush=True)

        if result["done"]:
            break

    return final_reward, steps_taken


def run_episode(task_id: str, email_index: int = 0) -> Tuple[float, int]:
    if ENV_MODE == "http":
        return run_episode_http(task_id, email_index)
    return run_episode_local(task_id, email_index)

# ---------------------------------------------------------------------------
# Main — [START]/[STEP]/[END] blocks with flush=True throughout
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set.", flush=True)

    if ENV_MODE == "http" and not SPACE_URL:
        print("ERROR: ENV_MODE=http requires SPACE_URL.", flush=True)
        sys.exit(1)

    task_scores: Dict[str, List[float]] = {
        TASK_CLASSIFY: [],
        TASK_PRIORITIZE: [],
        TASK_RESPOND: [],
    }

    for episode_cfg in TASKS:
        task_id = episode_cfg["task_id"]
        email_index = episode_cfg["email_index"]

        # [START] block — parsed by validator
        print(f"[START] task={task_id}", flush=True)

        reward, steps = run_episode(task_id, email_index)
        reward = _clamp(reward)
        task_scores[task_id].append(reward)

        # [END] block — parsed by validator
        print(f"[END] task={task_id} score={reward:.4f} steps={steps}", flush=True)

    # Summary
    all_scores = []
    for task_id, scores in task_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            all_scores.extend(scores)
            print(f"[SUMMARY] task={task_id} n={len(scores)} avg={avg:.4f}", flush=True)

    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"[SUMMARY] overall_avg={overall:.4f}", flush=True)


if __name__ == "__main__":
    main()