"""
Baseline inference script for the Email Triage OpenEnv environment.

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL   - LLM API endpoint (OpenAI-compatible)
    MODEL_NAME     - Model identifier
    HF_TOKEN / API_KEY - API key

Optional environment variables:
    ENV_MODE       - "local" (default) or "http"
                     "local"  — imports environment.py directly (all source
                                files must be co-located in the same directory)
                     "http"   — calls a deployed Space over HTTP; set
                                SPACE_URL to point at it
    SPACE_URL      - Base URL of the deployed HF Space, e.g.
                     https://your-username-email-triage-openenv.hf.space
                     Required only when ENV_MODE=http

The script runs the standard OpenAI client against all 3 tasks and prints
a reproducible baseline score report.

Task coverage:
  - email_classify  : all 10 labelled emails (email_index 0-9)
  - email_prioritize: 1 episode (5-email inbox)
  - email_respond   : all 3 complaint scenarios (email_index 0-2)
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

ENV_MODE: str = os.getenv("ENV_MODE", "local").lower()   # "local" | "http"
SPACE_URL: str = os.getenv("SPACE_URL", "").rstrip("/")  # e.g. https://user-repo.hf.space

MAX_STEPS = 10
TEMPERATURE = 0.0   # deterministic for reproducibility
MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# Mode: local — import environment directly (files must be co-located)
# ---------------------------------------------------------------------------

if ENV_MODE == "local":
    from environment import EmailTriageEnv, TASK_CLASSIFY, TASK_PRIORITIZE, TASK_RESPOND
    from models import Action, ActionType, EmailCategory
    from email_data import CLASSIFY_EMAILS, RESPONSE_EMAILS

    TASKS = (
        [{"task_id": TASK_CLASSIFY, "email_index": i} for i in range(len(CLASSIFY_EMAILS))]
        + [
            {"task_id": TASK_PRIORITIZE, "email_index": 0},
        ]
        + [{"task_id": TASK_RESPOND, "email_index": i} for i in range(len(RESPONSE_EMAILS))]
    )
else:
    # In HTTP mode these constants are referenced by name only
    TASK_CLASSIFY = "email_classify"
    TASK_PRIORITIZE = "email_prioritize"
    TASK_RESPOND = "email_respond"
    TASKS = (
        [{"task_id": TASK_CLASSIFY, "email_index": i} for i in range(10)]
        + [
            {"task_id": TASK_PRIORITIZE, "email_index": 0},
        ]
        + [{"task_id": TASK_RESPOND, "email_index": i} for i in range(3)]
    )

# ---------------------------------------------------------------------------
# HTTP client helpers (used when ENV_MODE == "http")
# ---------------------------------------------------------------------------

def _http_reset(task_id: str, email_index: int) -> Tuple[str, Dict[str, Any]]:
    """POST /reset -> (session_id, observation_dict)"""
    resp = requests.post(
        f"{SPACE_URL}/reset",
        json={"task_id": task_id, "email_index": email_index},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["session_id"], data["observation"]


def _http_step(session_id: str, action_dict: Dict[str, Any]) -> Dict[str, Any]:
    """POST /step -> step result dict"""
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
      reply that includes ALL of the following:
        * A greeting (Dear / Hello / Hi <name>)
        * Acknowledgment of the customer's specific issue
        * An apology for the inconvenience
        * A clear resolution (refund / replacement / account unlock as appropriate)
        * A request to verify account or order details
        * Empathetic, professional tone throughout
        * A professional sign-off (Kind regards / Sincerely / Best regards)
      The response must be at least 60 words.
    - Read the observation carefully — the goal tells you exactly what to do.
    """
).strip()


def build_user_prompt(observation: Dict[str, Any]) -> str:
    """Convert an observation dict into a user prompt for the LLM."""
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

    prompt = textwrap.dedent(
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
    return prompt


def parse_action(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the JSON action from the model response.
    Strips markdown fences if present.
    """
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Episode runners — one per mode
# ---------------------------------------------------------------------------

def run_episode_local(
    client: OpenAI,
    task_id: str,
    email_index: int = 0,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Run a single episode using the local environment import."""
    env = EmailTriageEnv(task_id=task_id, email_index=email_index)
    observation = env.reset()
    obs_dict = observation.model_dump()

    step_log = []
    final_reward = 0.0

    for step_num in range(1, MAX_STEPS + 1):
        if obs_dict.get("done", False):
            break

        user_prompt = build_user_prompt(obs_dict)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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

        action_dict = parse_action(response_text)
        if action_dict is None:
            print(f"    [Step {step_num}] Could not parse action. Using noop.")
            action_dict = {"action_type": "noop"}

        action = Action(
            action_type=ActionType(action_dict.get("action_type", "noop")),
            email_id=action_dict.get("email_id"),
            category=EmailCategory(action_dict["category"].lower())
                if action_dict.get("category") else None,
            ordered_ids=action_dict.get("ordered_ids"),
            response_text=action_dict.get("response_text"),
        )

        result = env.step(action)

        step_log.append(
            {
                "step": step_num,
                "action_type": action.action_type.value,
                "reward": result.reward,
                "done": result.done,
                "last_result": result.observation.last_action_result[:120],
            }
        )

        final_reward = max(final_reward, result.reward)
        obs_dict = result.observation.model_dump()

        if result.done:
            break

    return final_reward, step_log


def run_episode_http(
    client: OpenAI,
    task_id: str,
    email_index: int = 0,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Run a single episode against a deployed Space over HTTP."""
    session_id, obs_dict = _http_reset(task_id, email_index)

    step_log = []
    final_reward = 0.0

    for step_num in range(1, MAX_STEPS + 1):
        if obs_dict.get("done", False):
            break

        user_prompt = build_user_prompt(obs_dict)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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

        action_dict = parse_action(response_text)
        if action_dict is None:
            print(f"    [Step {step_num}] Could not parse action. Using noop.")
            action_dict = {"action_type": "noop"}

        result = _http_step(session_id, action_dict)
        obs_dict = result["observation"]
        reward = result["reward"]

        step_log.append(
            {
                "step": step_num,
                "action_type": action_dict.get("action_type", "noop"),
                "reward": reward,
                "done": result["done"],
                "last_result": obs_dict.get("last_action_result", "")[:120],
            }
        )

        final_reward = max(final_reward, reward)

        if result["done"]:
            break

    return final_reward, step_log


def run_episode(
    client: OpenAI,
    task_id: str,
    email_index: int = 0,
) -> Tuple[float, List[Dict[str, Any]]]:
    if ENV_MODE == "http":
        return run_episode_http(client, task_id, email_index)
    return run_episode_local(client, task_id, email_index)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Email Triage OpenEnv — Baseline Inference Script")
    print("=" * 60)
    print(f"  Model   : {MODEL_NAME}")
    print(f"  API URL : {API_BASE_URL}")
    print(f"  Mode    : {ENV_MODE}" + (f"  (Space: {SPACE_URL})" if ENV_MODE == "http" else ""))
    print()

    if not API_KEY:
        print("WARNING: No API key found. Set HF_TOKEN or API_KEY env var.")

    if ENV_MODE == "http" and not SPACE_URL:
        raise SystemExit(
            "ERROR: ENV_MODE=http requires SPACE_URL to be set.\n"
            "  e.g. export SPACE_URL=https://your-username-email-triage-openenv.hf.space"
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    task_scores: Dict[str, List[float]] = {
        TASK_CLASSIFY: [],
        TASK_PRIORITIZE: [],
        TASK_RESPOND: [],
    }

    for episode_cfg in TASKS:
        task_id = episode_cfg["task_id"]
        email_index = episode_cfg["email_index"]
        print(f"--- Task: {task_id}  (email_index={email_index}) ---")

        start = time.time()
        reward, step_log = run_episode(client, task_id, email_index)
        elapsed = time.time() - start

        task_scores[task_id].append(reward)

        for entry in step_log:
            print(
                f"  Step {entry['step']:>2}: [{entry['action_type']:<20}] "
                f"reward={entry['reward']:.3f}  done={entry['done']}  "
                f"| {entry['last_result'][:80]}"
            )
        print(f"  => Episode reward: {reward:.3f}  ({elapsed:.1f}s)\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  BASELINE SCORE REPORT")
    print("=" * 60)

    all_scores = []
    for task_id, scores in task_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            scores_str = ", ".join(f"{s:.3f}" for s in scores)
            n = len(scores)
            print(f"  {task_id:<25} n={n}  avg={avg:.3f}  scores=[{scores_str}]")
            all_scores.extend(scores)
        else:
            print(f"  {task_id:<25} (no episodes run)")

    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"\n  OVERALL AVERAGE: {overall:.3f}  (n={len(all_scores)} episodes)")

    print("=" * 60)


if __name__ == "__main__":
    main()