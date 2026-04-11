"""
inference.py
============
Baseline inference script for the SQL Data Quality Agent.

MANDATORY STDOUT FORMAT
-----------------------
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Rules:
  - One [START] line at episode begin.
  - One [STEP] line per step, immediately after env.step() returns.
  - One [END] line after the episode ends (always emitted, even on exception).
  - reward and rewards formatted to 2 decimal places.
  - done and success are lowercase booleans: true or false.
  - error is the raw last_action_result string, or null if none.
  - All fields on a single line with no newlines within a line.

Required environment variables:
  API_BASE_URL  - The API endpoint for the LLM.
  MODEL_NAME    - The model identifier for inference.
  HF_TOKEN      - Your Hugging Face / API key.

Usage:
    python inference.py
    ENV_SERVER_URL=http://localhost:7860 python inference.py
"""

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — all from environment variables as required
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY", "")
)
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_SERVER_URL: str = os.getenv("ENV_SERVER_URL", "http://localhost:7860")

BENCHMARK = "sql-data-quality-agent"
TASKS = ["null_patrol", "duplicate_destroyer", "constraint_cascade"]
MAX_STEPS = 12        # per task; keeps total runtime well under 20 min
TEMPERATURE = 0.1
MAX_TOKENS = 256

# ---------------------------------------------------------------------------
# OpenAI client (must use OpenAI Client as required)
# ---------------------------------------------------------------------------

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Score clamping — Phase 2 requires all scores strictly in (0, 1)
# ---------------------------------------------------------------------------

def clamp_val(v: float, low: float = 0.01, high: float = 0.99) -> float:
    """Clamp value to (0, 1) exclusive range. Ensures strictly between bounds."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.5
    # NaN / Inf
    if v != v or v == float('inf') or v == float('-inf'):
        return 0.5
    result = max(low, min(high, v))
    # Additional safety check for floating-point edge cases
    if result <= 0.0:
        result = low
    if result >= 1.0:
        result = high
    return round(result, 4)

# ---------------------------------------------------------------------------
# Mandatory stdout log helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    # Sanitise action: single line, no spaces that break parsing
    action_clean = action.replace("\n", " ").replace("\r", "").strip()
    error_val = error if error else "null"
    done_val = str(done).lower()
    clamped_reward = clamp_val(reward)
    print(
        f"[STEP] step={step} action={action_clean!r} "
        f"reward={clamped_reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    # Guard: if no steps taken, emit at least one valid reward
    if not rewards:
        rewards = [0.01]
    clamped_rewards = [clamp_val(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in clamped_rewards)
    success_val = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str, seed: int = 42) -> Dict:
    resp = requests.post(
        f"{ENV_SERVER_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(sql: str, rationale: str = "") -> Dict:
    resp = requests.post(
        f"{ENV_SERVER_URL}/step",
        json={"sql": sql, "rationale": rationale},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Data Quality Engineer working with SQLite databases.
At each turn you must output EXACTLY ONE SQL statement to improve data quality.

Rules:
- Output ONLY a raw SQL statement. No markdown, no explanation, no code fences.
- Valid statement types: UPDATE, DELETE, INSERT, SELECT.
- Do NOT use DROP TABLE or TRUNCATE.
- Use SELECT to inspect data when uncertain.
- Aim to maximise the quality score as quickly as possible.
""").strip()


def build_user_prompt(obs: Dict, step: int) -> str:
    report = obs["quality_report"]
    schema_str = json.dumps(obs["table_schema"], indent=2)
    samples = obs["sample_rows"]
    sample_parts = []
    for table, rows in samples.items():
        sample_parts.append(f"Table '{table}' (first {min(len(rows), 5)} rows):")
        sample_parts.append(json.dumps(rows[:5], indent=2))
    sample_str = "\n".join(sample_parts)
    hints = "\n".join(f"  - {h}" for h in obs.get("hints", []))

    return textwrap.dedent(f"""
TASK: {obs['task_description']}
STEP: {step}
LAST ACTION RESULT: {obs.get('last_action_result', 'N/A')}

QUALITY REPORT:
  Overall Score      : {report['overall_score']:.4f}
  Null Ratio         : {report.get('null_ratio', 0.0):.4f}
  Duplicate Ratio    : {report.get('duplicate_ratio', 0.0):.4f}
  Type Error Ratio   : {report.get('type_error_ratio', 0.0):.4f}
  FK Violation Ratio : {report.get('constraint_violation_ratio', 0.0):.4f}
  Value Error Ratio  : {report.get('value_error_ratio', 0.0):.4f}

DATABASE SCHEMA:
{schema_str}

SAMPLE DATA:
{sample_str}

HINTS:
{hints if hints else '  No hints available.'}

Output a single SQL statement now:
""").strip()

# ---------------------------------------------------------------------------
# Agent loop — one task episode
# ---------------------------------------------------------------------------

def run_task(task_id: str, seed: int = 42) -> None:
    """Run a full episode for one task, emitting mandatory stdout logs."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False

    try:
        obs = env_reset(task_id=task_id, seed=seed)

        TASK_THRESHOLDS = {
            "null_patrol": 0.85,
            "duplicate_destroyer": 0.85,
            "constraint_cascade": 0.80,
        }
        threshold = TASK_THRESHOLDS.get(task_id, 0.90)

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            user_prompt = build_user_prompt(obs, step)

            # LLM call
            sql = "SELECT 1"  # safe fallback
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                sql = response.choices[0].message.content.strip()
            except Exception as llm_err:
                sys.stderr.write(f"[WARN] LLM error at step {step}: {llm_err}\n")

            # Strip accidental markdown fences
            if sql.startswith("```"):
                sql = "\n".join(
                    line for line in sql.splitlines()
                    if not line.startswith("```")
                ).strip()

            # Execute action
            reward = 0.0
            done = False
            error: Optional[str] = None

            try:
                result = env_step(sql=sql, rationale=f"step {step}")
                obs = result["observation"]
                reward = clamp_val(result["reward"])
                done = result["done"]
                steps_taken = step
                rewards.append(reward)

                last_result = obs.get("last_action_result", "")
                if last_result and last_result != "success":
                    error = last_result

                # [STEP] log — emitted immediately after env.step()
                log_step(step=step, action=sql, reward=reward, done=done, error=error)

                if done:
                    final_score = clamp_val(obs["quality_report"]["overall_score"])
                    success = final_score >= threshold
                    break

            except Exception as step_err:
                error = str(step_err)
                rewards.append(clamp_val(0.0))
                log_step(step=step, action=sql, reward=clamp_val(0.0), done=True, error=error)
                steps_taken = step
                break

    except Exception as outer_err:
        sys.stderr.write(f"[ERROR] Task {task_id} failed: {outer_err}\n")
        success = False

    # [END] always emitted
    log_end(success=success, steps=steps_taken, rewards=rewards)


# ---------------------------------------------------------------------------
# Server readiness check
# ---------------------------------------------------------------------------

def wait_for_server(timeout: int = 60) -> None:
    """Poll /health until the server is ready."""
    for attempt in range(timeout):
        try:
            r = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
            if r.status_code == 200:
                sys.stderr.write(f"Server ready after {attempt}s\n")
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server at {ENV_SERVER_URL} not ready after {timeout}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        raise EnvironmentError(
            "No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY."
        )

    wait_for_server(timeout=60)

    for task_id in TASKS:
        run_task(task_id=task_id, seed=42)


if __name__ == "__main__":
    main()
