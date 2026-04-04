"""
FastAPI server exposing the OpenEnv HTTP interface.

Endpoints:
  POST /reset   — start a new episode
  POST /step    — take an action
  GET  /state   — inspect current state
  GET  /health  — liveness probe
  GET  /tasks   — list available tasks
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import Action, ActionType, EmailCategory
from environment import EmailTriageEnv, VALID_TASKS, TASK_CLASSIFY

app = FastAPI(
    title="Email Triage OpenEnv",
    description="OpenEnv-compliant email triage environment for AI agent evaluation.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Session store — one env per session_id, safe for concurrent users
#
# NOTE: Sessions are in-process memory only.
# They do NOT survive container restarts — callers must POST /reset again
# after a cold start. Sessions idle for longer than SESSION_TTL_SECONDS are
# evicted on the next /reset call to prevent unbounded memory growth.
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", 3600))  # 1 hour default

_sessions: Dict[str, EmailTriageEnv] = {}
_session_last_used: Dict[str, float] = {}


def _evict_stale_sessions() -> None:
    """Remove sessions that have not been touched within SESSION_TTL_SECONDS."""
    now = time.monotonic()
    stale = [
        sid
        for sid, last in _session_last_used.items()
        if now - last > SESSION_TTL_SECONDS
    ]
    for sid in stale:
        _sessions.pop(sid, None)
        _session_last_used.pop(sid, None)


def _get_env(session_id: str) -> EmailTriageEnv:
    if session_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown session_id '{session_id}'. Call POST /reset first.",
        )
    _session_last_used[session_id] = time.monotonic()
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = TASK_CLASSIFY
    email_index: Optional[int] = 0
    session_id: Optional[str] = None   # caller may supply their own ID


class StepRequest(BaseModel):
    session_id: str
    action_type: str
    email_id: Optional[str] = None
    category: Optional[str] = None
    ordered_ids: Optional[list] = None
    response_text: Optional[str] = None


class StateRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "email-triage-openenv"}


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "id": "email_classify",
                "name": "Email Classification",
                "difficulty": "easy",
                "description": "Classify a single email as urgent/normal/spam.",
            },
            {
                "id": "email_prioritize",
                "name": "Email Prioritization",
                "difficulty": "medium",
                "description": "Sort 5 inbox emails by priority (highest first).",
            },
            {
                "id": "email_respond",
                "name": "Email Response Drafting",
                "difficulty": "hard",
                "description": "Draft a professional response to a customer complaint email.",
            },
        ]
    }


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    # Evict stale sessions on each reset to keep memory bounded
    _evict_stale_sessions()

    task_id = req.task_id or TASK_CLASSIFY
    if task_id not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{task_id}'. Valid: {VALID_TASKS}",
        )

    session_id = req.session_id or str(uuid.uuid4())
    env = EmailTriageEnv(task_id=task_id, email_index=req.email_index or 0)
    observation = env.reset()
    _sessions[session_id] = env
    _session_last_used[session_id] = time.monotonic()

    return {
        "session_id": session_id,
        "observation": observation.model_dump(),
        "task_id": task_id,
        "message": "Episode reset successfully.",
    }


@app.post("/step")
async def step(req: StepRequest):
    env = _get_env(req.session_id)

    try:
        action_type = ActionType(req.action_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type '{req.action_type}'. "
                   f"Valid: {[a.value for a in ActionType]}",
        )

    category = None
    if req.category:
        try:
            category = EmailCategory(req.category.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category '{req.category}'. "
                       f"Valid: {[c.value for c in EmailCategory]}",
            )

    action = Action(
        action_type=action_type,
        email_id=req.email_id,
        category=category,
        ordered_ids=req.ordered_ids,
        response_text=req.response_text,
    )

    result = env.step(action)
    return {
        "session_id": req.session_id,
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
async def state(session_id: str):
    env = _get_env(session_id)
    return env.state()

# ---------------------------------------------------------------------------
# Multi-email episode grading endpoint
# ---------------------------------------------------------------------------

class ClassifyEpisodeRequest(BaseModel):
    classifications: Dict[str, str]  # {email_id: category}


@app.post("/grade/classify")
async def grade_classify_episode(req: ClassifyEpisodeRequest):
    """
    Grade an entire classify episode in one shot.
    Accepts a dict of {email_id: predicted_category} and returns
    accuracy + per-email breakdown. Does not require an active session.
    """
    from graders import grade_classification_episode
    reward, info = grade_classification_episode(req.classifications)
    return {"reward": reward, "info": info}

# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)