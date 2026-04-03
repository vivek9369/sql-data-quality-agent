"""
server/app.py
=============
OpenEnv-compliant server entry point.
Re-exports the FastAPI app from the root and provides a main() function
that openenv validate expects.
"""

import os
import sys

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Entry point for `uv run server` / `openenv serve`."""
    import uvicorn
    port = int(os.environ.get("PORT", port))
    print(f"Starting SQL Data Quality Agent on port {port} ...")
    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
