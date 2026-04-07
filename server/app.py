"""
server/app.py — OpenEnv multi-mode deployment entry point.

This file is required by `openenv validate` for multi-mode deployment.
It re-exports the FastAPI app from main.py so the openenv CLI can find it.
"""

import sys
import os

# Ensure the project root is on the path so main.py can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: F401  — re-export for openenv multi-mode

__all__ = ["app"]