"""
server/app.py — OpenEnv multi-mode deployment entry point.

Required by `openenv validate`. Exposes a main() function that launches
the FastAPI server via uvicorn.
"""

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Entry point for the OpenEnv server (called by openenv CLI)."""
    import uvicorn
    from main import app  # noqa: F401

    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()