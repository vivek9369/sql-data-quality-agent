---
title: SQL Data Quality Agent
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - data-quality
  - sql
  - reinforcement-learning
  - agent
---

# SQL Data Quality Agent — OpenEnv

> **An OpenEnv environment where an AI agent acts as a Data Quality Engineer.**
> Given a dirty SQLite database, the agent issues SQL statements to clean it — fixing NULLs, duplicates, type errors, and constraint violations.

---

## Table of Contents
1. [Environment Description](#environment-description)
2. [Why This Domain?](#why-this-domain)
3. [Action Space](#action-space)
4. [Observation Space](#observation-space)
5. [Reward Function](#reward-function)
6. [Tasks](#tasks)
7. [Setup & Usage](#setup--usage)
8. [Running the Baseline](#running-the-baseline)
9. [Baseline Scores](#baseline-scores)
10. [Deployment (HF Spaces + Docker)](#deployment)

---

## Environment Description

In real data pipelines, data engineers spend **60–80% of their time** cleaning dirty data: filling missing values, deduplicating records, fixing type mismatches, and resolving referential integrity violations.

This environment simulates that exact workflow. The agent is dropped into a SQLite database with deliberate data quality issues and must issue SQL statements (`UPDATE`, `DELETE`, `INSERT`, `SELECT`) to bring the dataset up to a target quality score.

The environment exposes a **dense reward signal** — every fixing step earns proportional credit, so the agent always has a learning signal even mid-episode.

---

## Why This Domain?

| Criterion | This Environment |
|---|---|
| **Real-world** | Data cleaning is a daily activity in every data team |
| **Novel in OpenEnv** | No existing SQL-level data quality environment exists |
| **Measurable** | Quality scores are fully deterministic and reproducible |
| **Scalable difficulty** | Same framework, 3 tasks from trivial to complex |

---

## Action Space

```json
{
  "sql": "UPDATE customers SET email = 'unknown@example.com' WHERE email IS NULL",
  "rationale": "Fill NULL emails with placeholder (optional, for logging)"
}
```

- **`sql`** (required): Any valid SQLite SQL statement. Errors are caught and returned as observations — the agent is never crashed.
- **`rationale`** (optional): The agent's reasoning. Logged but not executed.

---

## Observation Space

```json
{
  "task_id": "null_patrol",
  "task_description": "A customers table has ~20% NULL emails and phones...",
  "table_schema": {
    "customers": {"id": "INTEGER", "name": "TEXT", "email": "TEXT", "...": "..."}
  },
  "sample_rows": {
    "customers": [{"id": 1, "name": "Alice Smith", "email": null, "...": "..."}, "..."]
  },
  "quality_report": {
    "null_ratio": 0.19,
    "duplicate_ratio": 0.0,
    "type_error_ratio": 0.0,
    "constraint_violation_ratio": 0.0,
    "value_error_ratio": 0.0,
    "overall_score": 0.81
  },
  "last_action_result": "success",
  "step": 1,
  "done": false,
  "hints": ["UPDATE customers SET email = 'unknown@example.com' WHERE email IS NULL"]
}
```

---

## Reward Function

The reward is **dense** — the agent gets signal at every step:

| Component | Formula | Purpose |
|---|---|---|
| **Progress reward** | `score_delta × 10` | Main learning signal |
| **Milestone bonus** | `+0.05 – +0.30` | Bonus for large improvements |
| **Error penalty** | `-0.05` per SQL error | Discourages invalid SQL |
| **Destructive penalty** | `-0.5 / -1.0` | Penalises DROP TABLE, unguarded DELETE |
| **Efficiency bonus** | `+0.05 × steps_saved` | Rewards finishing early |

Range: approximately `[-1.0, +2.0]` per step.

---

## Tasks

### Task 1 — Null Patrol 🟢 Easy
- **Table**: `customers` (50 rows)
- **Issue**: ~20% of `email` and `phone` values are NULL
- **Goal**: Fill all NULLs with placeholder values
- **Success threshold**: `overall_score ≥ 0.95`
- **Max steps**: 15

### Task 2 — Duplicate Destroyer 🟡 Medium
- **Table**: `orders` (200 rows)
- **Issue**: ~15% are duplicate records (same `order_id`, multiple timestamps)
- **Goal**: Keep only the earliest row for each `order_id`
- **Success threshold**: `overall_score ≥ 0.95`
- **Max steps**: 20

### Task 3 — Constraint Cascade 🔴 Hard
- **Tables**: `products` + `inventory`
- **Issues** (3 categories):
  1. ~20% of `price` values stored as strings (`$12.99` instead of `12.99`)
  2. ~15% of `inventory` rows reference non-existent `product_id` (FK violations)
  3. ~10% of `quantity` values are negative
- **Goal**: Fix all three categories
- **Success threshold**: `overall_score ≥ 0.90`
- **Max steps**: 30

---

## Setup & Usage

### Prerequisites
- Python 3.11+
- Docker (for containerised deployment)

### Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the environment server
python app.py
# Server starts at http://localhost:7860
# Swagger docs at http://localhost:7860/docs

# 3. Verify it's running
curl http://localhost:7860/health
```

### API Quick-Start

```bash
# Reset to Task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "null_patrol", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"sql": "UPDATE customers SET email = '\''unknown@example.com'\'' WHERE email IS NULL"}'

# Check state
curl http://localhost:7860/state
```

### List Available Tasks

```bash
curl http://localhost:7860/tasks
```

---

## Running the Baseline

```bash
# Set credentials
export HF_TOKEN="your-hf-token"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"

# Start server in background
python app.py &

# Run inference (all 3 tasks)
python inference.py
```

Or with OpenAI directly:
```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"
export API_BASE_URL="https://api.openai.com/v1"
python inference.py
```

---

## Baseline Scores

Baseline agent: `Llama-3.3-70B-Instruct` via HF Inference API
Reproducible with `seed=42`. Full runs logged to `baseline_results.json`.

| Task | Difficulty | Initial Score | Final Score | Steps Used | Success |
|---|---|---|---|---|---|
| Null Patrol | Easy | 0.8400 | 1.0000 | 2 | ✓ |
| Duplicate Destroyer | Medium | 0.8500 | 1.0000 | 1 | ✓ |
| Constraint Cascade | Hard | 0.7810 | 0.9250 | 3 | ✓ |
| **Average** | — | **0.8237** | **0.9750** | — | — |

> **Note**: The expert (above) uses hint SQL directly. The LLM agent typically takes 2–12 steps per task.

### Running Validation

```bash
# Run the pre-submission validator (must pass before submitting)
python validate.py
```

Expected output: `VALIDATION SUMMARY: 37/37 checks passed`

---

## Deployment

### Docker

```bash
# Build
docker build -t sql-data-quality-agent .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN="your-token" \
  sql-data-quality-agent

# Test
curl http://localhost:7860/health
```

### Hugging Face Spaces

1. Create a new HF Space with **Docker** SDK
2. Push all files to the Space repository
3. Set Secrets: `HF_TOKEN`, `MODEL_NAME`, `API_BASE_URL`
4. The Space will auto-build and deploy (port 7860)

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes (inference) | LLM API endpoint |
| `MODEL_NAME` | Yes (inference) | Model identifier |
| `HF_TOKEN` | Yes (inference) | HF or API key |
| `PORT` | No | Server port (default: 7860) |
| `ENV_SERVER_URL` | No | Override server URL for inference.py |

---

## Project Structure

```
.
├── app.py              # FastAPI HTTP server (OpenEnv endpoints)
├── environment.py      # Core env class (reset/step/state + Pydantic models)
├── tasks.py            # Task registry + graders (3 tasks)
├── data_generator.py   # Dirty dataset factory
├── reward.py           # Dense reward shaping
├── inference.py        # Baseline agent (OpenAI client)
├── validate.py         # Pre-submission validation script (run before submitting!)
├── test_env.py         # Unit tests for all 3 tasks
├── openenv.yaml        # OpenEnv manifest
├── Dockerfile          # Container definition
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # This file
```

---

## Pre-Submission Checklist

Before submitting, run:

```bash
python validate.py
```

This checks:
- [x] `openenv.yaml` valid with all required fields
- [x] 3+ tasks with correct difficulty range (easy → hard)
- [x] All graders deterministic and produce scores in `[0.0, 1.0]`
- [x] `reset()` / `step()` / `state()` API works correctly
- [x] Reward function provides dense signal
- [x] Episodes terminate at success threshold
- [x] All required files present (`app.py`, `Dockerfile`, `inference.py`, etc.)

---

## License

MIT License — free to use, modify, and distribute.
