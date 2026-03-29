"""
Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

Usage:
    # Against live server (default, for HF Spaces deployment)
    python inference.py

    # Against local server
    ENV_SERVER_URL=http://localhost:7860 python inference.py
"""

import os
import json
import time
import textwrap
import requests
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

ENV_SERVER_URL: str = os.getenv("ENV_SERVER_URL", "http://localhost:7860")

MAX_STEPS = 15          # per task (well under 20-min runtime limit)
TEMPERATURE = 0.1
MAX_TOKENS = 256

TASKS = ["null_patrol", "duplicate_destroyer", "constraint_cascade"]

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
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


def env_state() -> Dict:
    resp = requests.get(f"{ENV_SERVER_URL}/state", timeout=10)
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Data Quality Engineer.
You are given access to a SQLite database with data quality issues.
At each turn, you must output EXACTLY ONE SQL statement to improve the data quality.

Rules:
- Output ONLY a raw SQL statement, no markdown, no explanation, no code fences.
- Valid statement types: UPDATE, DELETE, INSERT, SELECT.
- Do NOT use DROP TABLE or TRUNCATE.
- If you're unsure, use a SELECT to inspect data first.
- Aim to maximise the quality score as quickly as possible.

Your output must be a single SQL statement that can be fed directly to sqlite3.execute().
""").strip()


def build_user_prompt(obs: Dict, step: int) -> str:
    report = obs["quality_report"]
    schema_str = json.dumps(obs["table_schema"], indent=2)

    # Show limited sample rows to keep prompt small
    samples = obs["sample_rows"]
    sample_str_parts = []
    for table, rows in samples.items():
        sample_str_parts.append(f"Table '{table}' (first {len(rows)} rows):")
        sample_str_parts.append(json.dumps(rows[:10], indent=2))
    sample_str = "\n".join(sample_str_parts)

    hints = "\n".join(f"  - {h}" for h in obs.get("hints", []))

    return textwrap.dedent(f"""
TASK: {obs['task_description']}

STEP: {step}
LAST ACTION RESULT: {obs.get('last_action_result', 'N/A')}

QUALITY REPORT:
  Overall Score : {report['overall_score']:.4f}
  Null Ratio    : {report.get('null_ratio', 0.0):.4f}
  Duplicate Ratio: {report.get('duplicate_ratio', 0.0):.4f}
  Type Error Ratio: {report.get('type_error_ratio', 0.0):.4f}
  FK Violation Ratio: {report.get('constraint_violation_ratio', 0.0):.4f}
  Value Error Ratio: {report.get('value_error_ratio', 0.0):.4f}

DATABASE SCHEMA:
{schema_str}

SAMPLE DATA:
{sample_str}

HINTS (if you're stuck):
{hints if hints else "  No hints available."}

Output a single SQL statement now:
""").strip()

# ---------------------------------------------------------------------------
# Agent loop for one task
# ---------------------------------------------------------------------------

def run_task(task_id: str, seed: int = 42) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"TASK: {task_id.upper().replace('_', ' ')}")
    print(f"{'='*60}")

    obs = env_reset(task_id=task_id, seed=seed)
    initial_score = obs["quality_report"]["overall_score"]
    print(f"  Initial score: {initial_score:.4f}")

    total_reward = 0.0
    history: List[Dict] = []

    for step in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break

        user_prompt = build_user_prompt(obs, step)

        # LLM call
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
        except Exception as e:
            print(f"  [Step {step}] LLM error: {e}")
            sql = "SELECT 1"  # fallback no-op

        # Clean up any accidental markdown
        if sql.startswith("```"):
            sql = "\n".join(
                line for line in sql.splitlines()
                if not line.startswith("```")
            ).strip()

        print(f"  [Step {step}] SQL: {sql[:100]}{'...' if len(sql) > 100 else ''}")

        # Execute action
        try:
            result = env_step(sql=sql, rationale=f"step {step}")
            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]
            total_reward += reward

            score = obs["quality_report"]["overall_score"]
            print(f"           Score: {score:.4f}  Reward: {reward:+.4f}  Done: {done}")

            history.append({
                "step": step,
                "sql": sql,
                "score": score,
                "reward": reward,
                "result": obs.get("last_action_result", ""),
            })

            if done:
                break

        except Exception as e:
            print(f"  [Step {step}] Step error: {e}")
            break

    final_score = obs["quality_report"]["overall_score"]
    # Use task-specific thresholds: null_patrol & duplicate_destroyer → 0.95, constraint_cascade → 0.90
    TASK_THRESHOLDS = {"null_patrol": 0.95, "duplicate_destroyer": 0.95, "constraint_cascade": 0.90}
    threshold = TASK_THRESHOLDS.get(task_id, 0.90)
    success = obs.get("done", False) and final_score >= threshold

    print(f"\n  Final score : {final_score:.4f}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Steps used  : {len(history)}")
    print(f"  Success     : {'✓' if success else '✗'}")

    return {
        "task_id": task_id,
        "initial_score": initial_score,
        "final_score": final_score,
        "total_reward": round(total_reward, 4),
        "steps_used": len(history),
        "success": success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nSQL Data Quality Agent — Baseline Inference")
    print(f"Model  : {MODEL_NAME}")
    print(f"Server : {ENV_SERVER_URL}")
    print(f"Max steps per task: {MAX_STEPS}")

    if not API_KEY:
        raise EnvironmentError(
            "No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY."
        )

    # Wait for server to be ready (up to 30s)
    for attempt in range(30):
        try:
            r = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"\nServer ready ✓")
                break
        except Exception:
            pass
        print(f"  Waiting for server... ({attempt + 1}/30)")
        time.sleep(1)
    else:
        raise RuntimeError(f"Server at {ENV_SERVER_URL} did not respond in 30s.")

    results = []
    for task_id in TASKS:
        result = run_task(task_id=task_id, seed=42)
        results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<25} {'Initial':>8} {'Final':>8} {'Reward':>8} {'Steps':>6} {'OK':>4}")
    print("-" * 60)
    for r in results:
        ok = "✓" if r["success"] else "✗"
        print(
            f"{r['task_id']:<25} {r['initial_score']:>8.4f} {r['final_score']:>8.4f} "
            f"{r['total_reward']:>8.4f} {r['steps_used']:>6} {ok:>4}"
        )

    avg_final = sum(r["final_score"] for r in results) / len(results)
    print(f"\nAverage final score: {avg_final:.4f}")

    # Write results to JSON for reproducibility
    with open("baseline_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "results": results, "avg_score": avg_final}, f, indent=2)
    print("Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
