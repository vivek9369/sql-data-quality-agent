"""
app.py
======
FastAPI server exposing the SQL Data Quality Agent as an OpenEnv HTTP API.

Endpoints:
  GET  /health          -> health check (required for HF Space ping)
  GET  /tasks           -> list all available tasks
  POST /reset           -> start a new episode
  POST /step            -> take one action
  GET  /state           -> current environment state
  GET  /docs            -> auto-generated Swagger UI (FastAPI built-in)
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from environment import DataQualityEnv, DataQualityAction, DataQualityObservation, DataQualityState
from tasks import list_tasks, clamp_score, clamp_ratio

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SQL Data Quality Agent — OpenEnv",
    description=(
        "An OpenEnv environment where an AI agent acts as a Data Quality Engineer, "
        "fixing dirty SQL databases through iterative SQL statements. "
        "Supports 3 tasks: easy -> medium -> hard."
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
env = DataQualityEnv()


# ---------------------------------------------------------------------------
# Helper: recursively clamp all float values in a dict that look like scores
# ---------------------------------------------------------------------------

def _clamp_response_scores(obj: Any) -> Any:
    """Recursively walk a dict/list and clamp any float values that are
    exactly 0.0 or 1.0, or any float keys named *score*, *ratio*, *reward*."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(v, float):
                # Clamp any float field whose name suggests it's a score/ratio/reward
                key_lower = str(k).lower()
                if any(word in key_lower for word in ["score", "ratio", "reward"]):
                    result[k] = clamp_score(v) if "score" in key_lower or "reward" in key_lower else clamp_ratio(v)
                elif v <= 0.0 or v >= 1.0:
                    # For other floats that happen to be 0.0 or 1.0, leave them
                    # (they might be legitimate values like amounts, counts, etc.)
                    result[k] = v
                else:
                    result[k] = v
            else:
                result[k] = _clamp_response_scores(v)
        return result
    elif isinstance(obj, list):
        return [_clamp_response_scores(item) for item in obj]
    else:
        return obj


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "null_patrol"
    seed: int = 42

    class Config:
        extra = "ignore"


class StepRequest(BaseModel):
    sql: str
    rationale: str = ""


class StepResponse(BaseModel):
    observation: DataQualityObservation
    reward: float
    done: bool
    info: Dict[str, Any]

    @field_validator("reward", mode="before")
    @classmethod
    def _clamp_reward(cls, v: float) -> float:
        """Ensure reward is strictly in (0, 1) before serialization."""
        return clamp_score(v)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root():
    """Root endpoint — required by HF Spaces health probe."""
    return {"status": "healthy", "environment": "sql-data-quality-agent", "version": "1.0.0"}


@app.get("/health", tags=["meta"])
def health_check():
    """Health check — openenv validate expects {"status": "healthy"}."""
    return {"status": "healthy", "environment": "sql-data-quality-agent", "version": "1.0.0"}


@app.get("/metadata", tags=["meta"])
def metadata():
    """Metadata endpoint — openenv validate expects name + description."""
    return {
        "name": "sql-data-quality-agent",
        "description": (
            "An OpenEnv environment where an AI agent acts as a Data Quality Engineer. "
            "Given a dirty SQLite database (NULL values, duplicate rows, type errors, "
            "foreign-key violations), the agent issues SQL statements to bring the "
            "dataset to a target quality threshold."
        ),
        "version": "1.0.0",
        "author": "Vivek Kumar Maurya",
    }


@app.get("/schema", tags=["meta"])
def schema():
    """Schema endpoint — openenv validate expects action, observation, state JSON schemas."""
    return {
        "action": DataQualityAction.model_json_schema(),
        "observation": DataQualityObservation.model_json_schema(),
        "state": DataQualityState.model_json_schema(),
    }


@app.get("/tasks", tags=["meta"])
def get_tasks():
    """List all available tasks with metadata."""
    return {"tasks": list_tasks()}


@app.post("/reset", tags=["environment"])
async def reset(request: Request):
    """
    Start a new episode for the given task.
    Returns the initial observation (table schema, sample rows, quality report).
    Accepts: full JSON body, partial body, empty body, or NO body at all (uses defaults).
    """
    task_id = "null_patrol"
    seed = 42
    try:
        body = await request.body()
        if body:
            data = await request.json()
            task_id = data.get("task_id", task_id)
            seed = int(data.get("seed", seed))
    except Exception:
        pass  # No/invalid body — use defaults
    try:
        obs = env.reset(task_id=task_id, seed=seed)
        # Serialize and clamp all score/ratio/reward fields
        response_data = obs.model_dump()
        response_data = _clamp_response_scores(response_data)
        return JSONResponse(content=response_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/step", tags=["environment"])
def step(request: StepRequest):
    """
    Execute one SQL action.
    Returns the new observation, reward, done flag, and episode info.
    """
    try:
        action = DataQualityAction(sql=request.sql, rationale=request.rationale)
        obs, reward, done, info = env.step(action)
        # Final safety: clamp reward at the API boundary
        reward = clamp_score(reward)
        # Also clamp cumulative_reward in info
        if "cumulative_reward" in info:
            info["cumulative_reward"] = clamp_score(info["cumulative_reward"])
        # Serialize and clamp all score/ratio/reward fields
        response_data = {
            "observation": obs.model_dump(),
            "reward": clamp_score(reward),
            "done": done,
            "info": info,
        }
        response_data = _clamp_response_scores(response_data)
        return JSONResponse(content=response_data)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/state", tags=["environment"])
def state():
    """Return current episode metadata and state."""
    try:
        st = env.state()
        response_data = st.model_dump()
        response_data = _clamp_response_scores(response_data)
        return JSONResponse(content=response_data)
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
