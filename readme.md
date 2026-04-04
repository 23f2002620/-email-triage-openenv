---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - email
  - triage
  - agent-evaluation
  - reinforcement-learning
---

# 📧 Email Triage OpenEnv

A real-world **email triage environment** for training and evaluating AI agents.
Agents must classify, prioritize, and respond to realistic customer emails — tasks
that knowledge workers perform every day.

---

## 🎯 Motivation

Email triage is one of the most common knowledge-work tasks in the world.
Automating it well requires reading comprehension, business judgment, and
communication skills. This environment provides a rigorous, reproducible benchmark
for AI agents across three difficulty levels.

---

## 🗂️ Tasks

| Task ID | Name | Difficulty | Max Steps | Description |
|---|---|---|---|---|
| `email_classify` | Email Classification | Easy | 5 | Classify a single email as `urgent` / `normal` / `spam` |
| `email_prioritize` | Email Prioritization | Medium | 10 | Sort 5 inbox emails by priority (highest first) |
| `email_respond` | Email Response Drafting | Hard | 15 | Draft a professional response to a customer complaint |

### Task 1 — Email Classification (Easy)
The agent receives a single email and must classify it into one of three categories:
- **urgent** — production outages, security incidents, critical business failures
- **normal** — routine requests, meeting reschedules, standard queries
- **spam** — newsletters, phishing, promotional content

The pool contains 10 labelled emails (`email_index` 0–9) so episodes can be varied.

**Reward:** 1.0 for correct classification, 0.0 for incorrect.

### Task 2 — Email Prioritization (Medium)
The agent receives 5 emails of mixed urgency and must submit them in priority order
(most important first). The agent may resubmit within `max_steps` using feedback
from each attempt to refine its ordering.

**Reward:** Continuous 0.0–1.0 based on normalised Kendall tau similarity, with
bonuses for placing both urgent emails in the top 2 positions (+0.10) and spam at
the bottom (+0.05).

### Task 3 — Email Response Drafting (Hard)
The agent must draft a complete, professional response to a customer complaint.
Three distinct complaint scenarios are available (`email_index` 0–2):

| Index | Email ID | Scenario |
|---|---|---|
| 0 | `r001` | Duplicate billing charge |
| 1 | `r002` | Wrong item shipped |
| 2 | `r003` | Account locked |

Each response is scored against 7 weighted criteria:

| Criterion | Weight | Description |
|---|---|---|
| `has_greeting` | 10% | Opens with a greeting addressing the customer |
| `acknowledges_issue` | 20% | Acknowledges the specific complaint (per-scenario patterns) |
| `apologizes` | 15% | Contains an apology for the inconvenience |
| `mentions_resolution` | 25% | Mentions the appropriate resolution (refund / replacement / unlock) |
| `requests_verification` | 10% | Asks for account or order details to verify |
| `professional_tone` | 10% | Empathetic, professional language throughout |
| `has_signoff` | 10% | Ends with a professional sign-off |

**Reward:** Weighted sum of criteria met (0.0–1.0), with a small length penalty
for responses under 50 words.

---

## 📐 OpenEnv Interface

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok"}` |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Take an action |
| `GET` | `/state` | Inspect full episode state |
| `GET` | `/tasks` | List available tasks |

### POST /reset

```json
{
  "task_id": "email_classify",
  "email_index": 0
}
```

Returns an `Observation` object:

```json
{
  "goal": "Classify the email (id=e001) into one of: urgent, normal, spam...",
  "emails": [{"id": "e001", "subject": "...", "body": "...", "sender": "...", "timestamp": "..."}],
  "inbox_state": {"classified": {}, "remaining": ["e001"]},
  "step": 0,
  "last_action_result": "",
  "last_action_error": false,
  "done": false
}
```

Also returns `session_id` — pass this in every subsequent `/step` and `/state` call.

### POST /step

**Classify an email:**
```json
{
  "session_id": "<session_id>",
  "action_type": "classify_email",
  "email_id": "e001",
  "category": "urgent"
}
```

**Prioritize inbox:**
```json
{
  "session_id": "<session_id>",
  "action_type": "prioritize_inbox",
  "ordered_ids": ["p001", "p003", "p002", "p005", "p004"]
}
```

**Draft a response:**
```json
{
  "session_id": "<session_id>",
  "action_type": "draft_response",
  "email_id": "r001",
  "response_text": "Dear Jane, we apologise for the duplicate charge..."
}
```

**No-op:**
```json
{"session_id": "<session_id>", "action_type": "noop"}
```

Returns:
```json
{
  "session_id": "<session_id>",
  "observation": {"goal": "...", "step": 1, "done": false, ...},
  "reward": 0.95,
  "done": false,
  "info": {"criteria": {...}}
}
```

### GET /state

```
GET /state?session_id=<session_id>
```

Returns full episode state including cumulative reward, history, and task-specific fields.

---

## 🏗️ Project Structure

```
.
├── Dockerfile           # Container build
├── requirements.txt     # Python dependencies
├── openenv.yaml         # Environment metadata (OpenEnv spec)
├── README.md            # This file
├── main.py              # FastAPI server (reset/step/state/health endpoints)
├── environment.py       # Core environment logic (step/reset/state)
├── models.py            # Typed Pydantic models (Observation, Action, StepResult)
├── email_data.py        # Deterministic email fixtures + response criteria
├── graders.py           # Task graders (deterministic, reproducible, per-scenario)
├── inference.py         # Baseline inference script (OpenAI client)
└── tests/
    └── test_all.py      # 45 unit tests covering graders + environment
```

---

## 🚀 Setup & Usage

### Local (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
# Server runs at http://localhost:7860

# Run baseline inference (requires API credentials)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

### Docker

```bash
# Build
docker build -t email-triage-openenv .

# Run
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="hf_..." \
  email-triage-openenv
```

### Quick API test

```bash
# Health check
curl http://localhost:7860/health

# Reset to email classification task
RESP=$(curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "email_classify", "email_index": 0}')
SESSION=$(echo $RESP | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# Classify the email
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"action_type\": \"classify_email\", \"email_id\": \"e001\", \"category\": \"urgent\"}"
```

---

## 📊 Baseline Scores

Measured with `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace inference router
(14 episodes total: 10 classify + 1 prioritize + 3 respond):

| Task | Episodes | Avg Score |
|---|---|---|
| `email_classify` | 10 | ~0.90 |
| `email_prioritize` | 1 | ~0.65 |
| `email_respond` | 3 | ~0.55 |
| **Overall** | **14** | **~0.75** |

Run `python inference.py` to reproduce these scores.

---

## ⚙️ Environment Variables

| Variable | Description | Required |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) | Yes |
| `MODEL_NAME` | Model identifier string | Yes |
| `HF_TOKEN` | HuggingFace API key | Yes (or `API_KEY`) |
| `API_KEY` | Alternative API key env var | Yes (or `HF_TOKEN`) |
| `ENV_MODE` | `local` (default) or `http` | No |
| `SPACE_URL` | Base URL of deployed HF Space (HTTP mode only) | No |
| `PORT` | Server port (default: 7860) | No |

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
# 45 tests — graders + environment + edge cases for all 3 email scenarios
```

---

## 📋 Action Space

| Action | Required Fields | Description |
|---|---|---|
| `classify_email` | `email_id`, `category` | Classify as `urgent` / `normal` / `spam` |
| `prioritize_inbox` | `ordered_ids` | List of email IDs, highest priority first |
| `draft_response` | `email_id`, `response_text` | Full professional reply text |
| `noop` | — | Take no action (step is consumed) |

## 📋 Observation Space

| Field | Type | Description |
|---|---|---|
| `goal` | string | Episode objective in plain English |
| `emails` | list[Email] | Emails to process (id, subject, body, sender, timestamp) |
| `inbox_state` | object | Current processing state (task-specific) |
| `step` | int | Current step number (0-indexed at reset) |
| `last_action_result` | string | Human-readable feedback from the last action |
| `last_action_error` | bool | `true` if the last action produced a validation error |
| `done` | bool | `true` when the episode has ended |