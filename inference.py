"""
Baseline inference script for the Email Triage OpenEnv environment.

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL   - LLM API endpoint (OpenAI-compatible)
    MODEL_NAME     - Model identifier to use for inference
    HF_TOKEN       - Your Hugging Face / API key

The script runs the standard OpenAI client against all 3 tasks and prints
structured logs (START/STEP/END) for each episode.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# Defaults are set only for API_BASE_URL and MODEL_NAME (not HF_TOKEN)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "<your-api-base-url>")
MODEL_NAME: str = os.getenv("MODEL_NAME", "<your-action-model-name>")
HF_TOKEN: str = os.getenv("HF_TOKEN")

# Optional — only needed when ENV_MODE=http
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME")

ENV_MODE: str = os.getenv("ENV_MODE", "local").lower()   # "local" | "http"
SPACE_URL: str = (os.getenv("SPACE_URL") or "").rstrip("/")

MAX_STEPS = 10
TEMPERATURE = 0.0   # deterministic for reproducibility
MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# All LLM calls use the OpenAI client configured via these variables
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

# ---------------------------------------------------------------------------
# Local environment import (ENV_MODE=local)
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
# HTTP helpers (ENV_MODE=http)
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
    payload = {"session_id": session_id, **action_dict}
    resp = requests.post(f"{SPACE_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI email triage assistant operating inside an OpenEnv environment.
    You will be given an observation describing your current task and the emails
    available in the inbox.

    You must respond with a single valid JSON action object. The action must
    follow exactly one of these schemas:

    1. Classify an email:
       {"action_type": "classify_email", "email_id": "<id>", "category": "urgent"|"normal"|"spam"}

    2. Prioritize the inbox (list email IDs highest priority first):
       {"action_type": "prioritize_inbox", "ordered_ids": ["<id1>", "<id2>", ...]}

    3. Draft a response to an email:
       {"action_type": "draft_response", "email_id": "<id>", "response_text": "<your full response>"}

    4. Do nothing:
       {"action_type": "noop"}

    RULES:
    - Output ONLY valid JSON — no prose, no markdown fences, no explanation.
    - For classify_email: category must be exactly "urgent", "normal", or "spam".
    - For draft_response: response_text must be a complete, professional email
      reply (greeting, acknowledgment, apology, resolution, sign-off).
    - Read the observation carefully — the goal tells you exactly what to do.
    """
).strip()


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
    inbox = observation.get("inbox_state", {})
    last_result = observation.get("last_action_result", "")
    last_error = observation.get("last_action_error", False)

    return textwrap.dedent(
        f"""
        GOAL: {observation.get('goal', '(none)')}

        STEP: {observation.get('step', 0)}

        EMAILS IN INBOX:
        {emails_text}

        INBOX STATE: {json.dumps(inbox, indent=2)}

        LAST ACTION RESULT: {last_result or '(none)'}
        LAST ACTION ERROR: {last_error}

        Respond with a single valid JSON action.
        """
    ).strip()


def parse_action(response_text: str) -> Optional[Dict[str, Any]]:
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
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

def run_episode_local(task_id: str, email_index: int = 0) -> Tuple[float, List[Dict]]:
    env = EmailTriageEnv(task_id=task_id, email_index=email_index)
    observation = env.reset()
    obs_dict = observation.model_dump()
    step_log = []
    final_reward = 0.0

    for step_num in range(1, MAX_STEPS + 1):
        if obs_dict.get("done", False):
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(obs_dict)},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"    [Step {step_num}] API error: {exc}. Using noop.")
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
        reward = result.reward

        # Stdout logs follow the required structured format (START/STEP/END)
        print(f"STEP {step_num} | task={task_id} | action={action.action_type.value} | reward={reward:.4f} | done={result.done}")

        step_log.append({
            "step": step_num,
            "action_type": action.action_type.value,
            "reward": reward,
            "done": result.done,
            "last_result": result.observation.last_action_result[:120],
        })

        final_reward = max(final_reward, reward)
        obs_dict = result.observation.model_dump()

        if result.done:
            break

    return final_reward, step_log


def run_episode_http(task_id: str, email_index: int = 0) -> Tuple[float, List[Dict]]:
    session_id, obs_dict = _http_reset(task_id, email_index)
    step_log = []
    final_reward = 0.0

    for step_num in range(1, MAX_STEPS + 1):
        if obs_dict.get("done", False):
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(obs_dict)},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"    [Step {step_num}] API error: {exc}. Using noop.")
            response_text = '{"action_type": "noop"}'

        action_dict = parse_action(response_text) or {"action_type": "noop"}
        result = _http_step(session_id, action_dict)
        obs_dict = result["observation"]
        reward = result["reward"]

        # Stdout logs follow the required structured format (START/STEP/END)
        print(f"STEP {step_num} | task={task_id} | action={action_dict.get('action_type','noop')} | reward={reward:.4f} | done={result['done']}")

        step_log.append({
            "step": step_num,
            "action_type": action_dict.get("action_type", "noop"),
            "reward": reward,
            "done": result["done"],
            "last_result": obs_dict.get("last_action_result", "")[:120],
        })

        final_reward = max(final_reward, reward)
        if result["done"]:
            break

    return final_reward, step_log


def run_episode(task_id: str, email_index: int = 0) -> Tuple[float, List[Dict]]:
    if ENV_MODE == "http":
        return run_episode_http(task_id, email_index)
    return run_episode_local(task_id, email_index)

# ---------------------------------------------------------------------------
# Main — structured START/STEP/END stdout format
# ---------------------------------------------------------------------------

def main() -> None:
    # START log
    print("START")
    print(f"model={MODEL_NAME}")
    print(f"api_base={API_BASE_URL}")
    print(f"mode={ENV_MODE}")
    print()

    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Set it via: export HF_TOKEN=hf_...")

    if ENV_MODE == "http" and not SPACE_URL:
        raise SystemExit(
            "ERROR: ENV_MODE=http requires SPACE_URL.\n"
            "  export SPACE_URL=https://your-username-email-triage-openenv.hf.space"
        )

    task_scores: Dict[str, List[float]] = {
        TASK_CLASSIFY: [],
        TASK_PRIORITIZE: [],
        TASK_RESPOND: [],
    }

    for episode_cfg in TASKS:
        task_id = episode_cfg["task_id"]
        email_index = episode_cfg["email_index"]

        # START episode
        print(f"START_EPISODE task={task_id} email_index={email_index}")
        start = time.time()

        reward, step_log = run_episode(task_id, email_index)
        elapsed = time.time() - start

        task_scores[task_id].append(reward)

        # END episode
        print(f"END_EPISODE task={task_id} email_index={email_index} reward={reward:.4f} elapsed={elapsed:.1f}s")
        print()

    # END summary
    print("END")
    print()
    print("=" * 50)
    print("BASELINE SCORE REPORT")
    print("=" * 50)

    all_scores = []
    for task_id, scores in task_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            scores_str = ", ".join(f"{s:.4f}" for s in scores)
            print(f"  {task_id:<25} n={len(scores)}  avg={avg:.4f}  scores=[{scores_str}]")
            all_scores.extend(scores)
        else:
            print(f"  {task_id:<25} (no episodes run)")

    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"\n  OVERALL AVERAGE: {overall:.4f}")

    print("=" * 50)


if __name__ == "__main__":
    main()