"""
environment.py
==============
Core OpenEnv environment class for the SQL Data Quality Agent.

Implements the standard OpenEnv interface:
  - reset(task_id, seed) → DataQualityObservation
  - step(action) → (DataQualityObservation, reward, done, info)
  - state() → DataQualityState

All models are typed Pydantic BaseModels for spec compliance.
"""

import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from tasks import TASK_REGISTRY, QualityReport, get_task, clamp_score
from data_generator import TASK_DB_GENERATORS
from reward import compute_reward


# ---------------------------------------------------------------------------
# Typed Pydantic Models (OpenEnv spec requirement)
# ---------------------------------------------------------------------------

class DataQualityAction(BaseModel):
    """Action: a single SQL statement + optional agent rationale."""
    sql: str = Field(..., description="SQL statement to execute (SELECT/UPDATE/DELETE/INSERT)")
    rationale: str = Field(default="", description="Agent's reasoning for this action")


class DataQualityObservation(BaseModel):
    """Observation returned after reset() or step()."""
    task_id: str
    task_description: str
    table_schema: Dict[str, Dict[str, str]] = Field(
        description="Table name → {column: type}"
    )
    sample_rows: Dict[str, List[Dict[str, Any]]] = Field(
        description="Table name → list of up to 20 sample rows"
    )
    quality_report: QualityReport
    last_action_result: str = Field(default="", description="'success' or error message")
    step: int = 0
    done: bool = False
    hints: List[str] = Field(default_factory=list)


class DataQualityState(BaseModel):
    """Full internal state (for the /state endpoint)."""
    episode_id: str
    task_id: str
    step: int
    max_steps: int
    current_score: float
    cumulative_reward: float
    tables: List[str]
    db_row_counts: Dict[str, int]


# ---------------------------------------------------------------------------
# Main Environment Class
# ---------------------------------------------------------------------------

class DataQualityEnv:
    """
    SQL Data Quality Agent Environment.

    The agent receives a dirty SQLite database and must issue SQL statements
    to improve its data quality score (0.0 → 1.0).
    """

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        self._task_id: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._step: int = 0
        self._max_steps: int = 0
        self._prev_score: float = 0.0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._seed: int = 42

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "null_patrol", seed: int = 42) -> DataQualityObservation:
        """
        Initialise a fresh episode for the given task.
        Returns the initial observation.
        """
        task_data = get_task(task_id)
        if task_data is None:
            raise ValueError(f"Unknown task_id: '{task_id}'. Valid tasks: {list(TASK_REGISTRY.keys())}")

        # Build a fresh dirty database
        generator = TASK_DB_GENERATORS[task_id]
        self._conn = generator(seed=seed)
        self._task_id = task_id
        self._episode_id = str(uuid.uuid4())
        self._step = 0
        self._max_steps = task_data["meta"].max_steps
        self._done = False
        self._cumulative_reward = 0.0
        self._seed = seed

        # Initial quality score
        report = task_data["grader"](self._conn)
        report.overall_score = clamp_score(report.overall_score)
        self._prev_score = report.overall_score

        return self._build_observation(report, last_result="", task_data=task_data)

    def step(self, action: DataQualityAction) -> Tuple[DataQualityObservation, float, bool, Dict]:
        """
        Execute one SQL action, compute reward, and return (obs, reward, done, info).
        """
        if self._conn is None or self._done:
            raise RuntimeError("Call reset() before step(), or episode is already done.")

        task_data = get_task(self._task_id)
        self._step += 1

        # Execute SQL
        result, last_result = self._execute_sql(action.sql)

        # Compute new quality score
        report = task_data["grader"](self._conn)
        curr_score = clamp_score(report.overall_score)
        report.overall_score = curr_score  # ensure observation also has clamped value

        # Compute reward
        reward = compute_reward(
            prev_score=self._prev_score,
            curr_score=curr_score,
            sql=action.sql,
            action_result=last_result,
            step=self._step,
            max_steps=self._max_steps,
            success_threshold=task_data["meta"].success_threshold,
        )
        # Clamp reward to (0, 1) exclusive — Phase 2 requirement
        reward = clamp_score(reward)
        self._cumulative_reward += reward
        self._prev_score = curr_score

        # Check done conditions
        threshold = task_data["meta"].success_threshold
        self._done = (curr_score >= threshold) or (self._step >= self._max_steps)

        obs = self._build_observation(report, last_result=last_result, task_data=task_data)
        info = {
            "episode_id": self._episode_id,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "step": self._step,
            "success": curr_score >= threshold,
            "sql_result": result,
        }
        return obs, reward, self._done, info

    def state(self) -> DataQualityState:
        """Return the current non-observation state (episode metadata)."""
        if self._conn is None:
            raise RuntimeError("Call reset() first.")

        cur = self._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]

        row_counts = {}
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM '{table}'")
            row_counts[table] = cur.fetchone()[0]

        return DataQualityState(
            episode_id=self._episode_id,
            task_id=self._task_id,
            step=self._step,
            max_steps=self._max_steps,
            current_score=clamp_score(self._prev_score),
            cumulative_reward=clamp_score(round(self._cumulative_reward, 4)),
            tables=tables,
            db_row_counts=row_counts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_sql(self, sql: str) -> Tuple[Any, str]:
        """Execute a SQL statement. Returns (result, status_string)."""
        cur = self._conn.cursor()
        try:
            cur.execute(sql)
            self._conn.commit()
            rows = cur.fetchall()
            if rows:
                result = [dict(row) for row in rows]
            else:
                result = f"OK ({cur.rowcount} rows affected)"
            return result, "success"
        except sqlite3.Error as e:
            return None, f"error: {str(e)}"
        except Exception as e:
            return None, f"error: {str(e)}"

    def _get_schema(self) -> Dict[str, Dict[str, str]]:
        """Return table schemas as {table: {col: type}}."""
        cur = self._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]

        schema = {}
        for table in tables:
            cur.execute(f"PRAGMA table_info('{table}')")
            schema[table] = {row[1]: row[2] for row in cur.fetchall()}
        return schema

    def _get_samples(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return up to 20 sample rows per table."""
        cur = self._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]

        samples = {}
        for table in tables:
            cur.execute(f"SELECT * FROM '{table}' LIMIT 20")
            samples[table] = [dict(row) for row in cur.fetchall()]
        return samples

    def _build_observation(
        self,
        report: QualityReport,
        last_result: str,
        task_data: Dict,
    ) -> DataQualityObservation:
        return DataQualityObservation(
            task_id=self._task_id,
            task_description=task_data["meta"].description,
            table_schema=self._get_schema(),
            sample_rows=self._get_samples(),
            quality_report=report,
            last_action_result=last_result,
            step=self._step,
            done=self._done,
            hints=task_data.get("hints", []),
        )
