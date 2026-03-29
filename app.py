"""
app.py
======
FastAPI server exposing the SQL Data Quality Agent as an OpenEnv HTTP API.

Endpoints:
  GET  /health          → health check (required for HF Space ping)
  GET  /tasks           → list all available tasks
  POST /reset           → start a new episode
  POST /step            → take one action
  GET  /state           → current environment state
  GET  /docs            → auto-generated Swagger UI (FastAPI built-in)
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import DataQualityEnv, DataQualityAction, DataQualityObservation, DataQualityState
from tasks import list_tasks

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SQL Data Quality Agent — OpenEnv",
    description=(
        "An OpenEnv environment where an AI agent acts as a Data Quality Engineer, "
        "fixing dirty SQL databases through iterative SQL statements. "
        "Supports 3 tasks: easy → medium → hard."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (stateful per-container)
# For production multi-agent use, this would be session-keyed.
env = DataQualityEnv()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "null_patrol"
    seed: int = 42


class StepRequest(BaseModel):
    sql: str
    rationale: str = ""


class StepResponse(BaseModel):
    observation: DataQualityObservation
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health_check():
    """Health check — must return 200 for HF Space validation."""
    return {"status": "ok", "environment": "sql-data-quality-agent", "version": "1.0.0"}


@app.get("/tasks", tags=["meta"])
def get_tasks():
    """List all available tasks with metadata."""
    return {"tasks": list_tasks()}


@app.post("/reset", response_model=DataQualityObservation, tags=["environment"])
def reset(request: ResetRequest):
    """
    Start a new episode for the given task.
    Returns the initial observation (table schema, sample rows, quality report).
    """
    try:
        obs = env.reset(task_id=request.task_id, seed=request.seed)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/step", response_model=StepResponse, tags=["environment"])
def step(request: StepRequest):
    """
    Execute one SQL action.
    Returns the new observation, reward, done flag, and episode info.
    """
    try:
        action = DataQualityAction(sql=request.sql, rationale=request.rationale)
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/state", response_model=DataQualityState, tags=["environment"])
def state():
    """Return current episode metadata and state."""
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting SQL Data Quality Agent on port {port} ...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
