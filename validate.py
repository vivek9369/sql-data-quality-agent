"""
validate.py
===========
Pre-submission validation script for the SQL Data Quality Agent OpenEnv environment.

Checks all requirements from the hackathon pre-submission checklist:
  1. openenv.yaml is valid and has required fields
  2. All 3 tasks exist with graders that produce scores in [0.0, 1.0]
  3. Graders are deterministic (reproducible with same seed)
  4. reset() / step() / state() all work correctly
  5. Reward function produces valid signals
  6. Episode terminates correctly at success threshold

Usage:
    python validate.py

All checks must pass before submitting. Exit code 0 = all pass.
"""

import sys
import yaml
import traceback
from typing import List, Tuple

# Ensure we can run from any directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results: List[Tuple[bool, str, str]] = []


def check(name: str, fn):
    """Run a single check and record result."""
    try:
        msg = fn()
        results.append((True, name, msg or "OK"))
        print(f"  {PASS} {name}: {msg or 'OK'}")
    except AssertionError as e:
        results.append((False, name, str(e)))
        print(f"  {FAIL} {name}: {e}")
    except Exception as e:
        results.append((False, name, f"Exception: {e}"))
        print(f"  {FAIL} {name}: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# 1. openenv.yaml validation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Validating openenv.yaml ...")

def check_yaml_exists():
    assert os.path.exists("openenv.yaml"), "openenv.yaml not found in project root"
    return "file exists"

def check_yaml_parses():
    with open("openenv.yaml", "r") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "openenv.yaml must be a YAML mapping"
    return f"{len(data)} top-level keys"

def check_yaml_required_fields():
    with open("openenv.yaml", "r") as f:
        data = yaml.safe_load(f)
    for field in ["name", "version", "description", "tasks", "endpoints",
                  "observation_space", "action_space", "reward_function"]:
        assert field in data, f"Missing required field: '{field}'"
    return "all required fields present"

def check_yaml_tasks():
    with open("openenv.yaml", "r") as f:
        data = yaml.safe_load(f)
    tasks = data.get("tasks", [])
    assert len(tasks) >= 3, f"Need at least 3 tasks, found {len(tasks)}"
    difficulties = {t["difficulty"] for t in tasks}
    assert "easy" in difficulties, "Need at least one 'easy' task"
    assert "hard" in difficulties, "Need at least one 'hard' task"
    return f"{len(tasks)} tasks: {[t['id'] for t in tasks]}"

def check_yaml_endpoints():
    with open("openenv.yaml", "r") as f:
        data = yaml.safe_load(f)
    eps = data.get("endpoints", {})
    for ep in ["reset", "step", "state"]:
        assert ep in eps, f"Missing required endpoint: '{ep}'"
    return f"endpoints: {list(eps.keys())}"

check("openenv.yaml exists", check_yaml_exists)
check("openenv.yaml parses", check_yaml_parses)
check("required fields present", check_yaml_required_fields)
check("3+ tasks with difficulties", check_yaml_tasks)
check("required endpoints defined", check_yaml_endpoints)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Task registry
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Validating task registry (tasks.py) ...")

from tasks import TASK_REGISTRY, list_tasks

def check_task_count():
    assert len(TASK_REGISTRY) >= 3, f"Need at least 3 tasks, got {len(TASK_REGISTRY)}"
    return f"{len(TASK_REGISTRY)} tasks registered"

def check_task_fields():
    for tid, tdata in TASK_REGISTRY.items():
        meta = tdata["meta"]
        assert meta.task_id == tid, f"task_id mismatch: {meta.task_id} vs {tid}"
        assert meta.difficulty in ("easy", "medium", "hard"), \
            f"Invalid difficulty '{meta.difficulty}' for {tid}"
        assert 0.0 < meta.success_threshold <= 1.0, \
            f"Invalid success_threshold {meta.success_threshold} for {tid}"
        assert meta.max_steps > 0, f"max_steps must be > 0 for {tid}"
        assert callable(tdata.get("grader")), f"Missing grader for {tid}"
    return "all task fields valid"

def check_list_tasks():
    tasks = list_tasks()
    assert len(tasks) == len(TASK_REGISTRY)
    return f"list_tasks() returns {len(tasks)} items"

check("at least 3 tasks", check_task_count)
check("task fields valid", check_task_fields)
check("list_tasks() works", check_list_tasks)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Environment API (reset / step / state)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Validating environment API ...")

from environment import DataQualityEnv, DataQualityAction, DataQualityObservation, DataQualityState

def check_reset_returns_obs():
    env = DataQualityEnv()
    obs = env.reset("null_patrol", seed=42)
    assert isinstance(obs, DataQualityObservation), "reset() must return DataQualityObservation"
    assert obs.task_id == "null_patrol"
    assert isinstance(obs.table_schema, dict)
    assert isinstance(obs.sample_rows, dict)
    score = obs.quality_report.overall_score
    assert 0.0 <= score <= 1.0, f"Initial score {score} out of [0,1]"
    return f"initial score={score:.4f}"

def check_step_returns_tuple():
    env = DataQualityEnv()
    env.reset("null_patrol", seed=42)
    obs, reward, done, info = env.step(DataQualityAction(sql="SELECT 1"))
    assert isinstance(obs, DataQualityObservation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "episode_id" in info
    return f"step() OK, reward={reward:+.4f}, done={done}"

def check_state_returns_state():
    env = DataQualityEnv()
    env.reset("null_patrol", seed=42)
    st = env.state()
    assert isinstance(st, DataQualityState)
    assert st.task_id == "null_patrol"
    assert st.step == 0
    assert st.episode_id is not None
    return f"episode_id={st.episode_id[:8]}..."

def check_reset_produces_clean_state():
    env = DataQualityEnv()
    env.reset("null_patrol", seed=42)
    env.step(DataQualityAction(sql="UPDATE customers SET email = 'x@x.com' WHERE email IS NULL"))
    # Reset again — must produce original dirty state
    obs2 = env.reset("null_patrol", seed=42)
    assert obs2.step == 0, "After reset(), step must be 0"
    score2 = obs2.quality_report.overall_score
    assert score2 < 0.98, f"After reset(), score should reflect dirty data, got {score2}"
    return f"re-reset score={score2:.4f} (clean dirty state)"

check("reset() returns DataQualityObservation", check_reset_returns_obs)
check("step() returns (obs, reward, done, info)", check_step_returns_tuple)
check("state() returns DataQualityState", check_state_returns_state)
check("reset() produces clean dirty state", check_reset_produces_clean_state)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grader validation (all 3 tasks)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Validating graders for all 3 tasks ...")

def check_grader_scores_in_range(task_id: str):
    def _check():
        env = DataQualityEnv()
        obs = env.reset(task_id, seed=42)
        score = obs.quality_report.overall_score
        assert 0.0 <= score <= 1.0, f"Initial score {score} out of [0.0, 1.0]"
        return f"initial score={score:.4f}"
    return _check

def check_grader_deterministic(task_id: str):
    def _check():
        env_a = DataQualityEnv()
        env_b = DataQualityEnv()
        obs_a = env_a.reset(task_id, seed=77)
        obs_b = env_b.reset(task_id, seed=77)
        s_a = obs_a.quality_report.overall_score
        s_b = obs_b.quality_report.overall_score
        assert s_a == s_b, f"Non-deterministic: {s_a} vs {s_b}"
        return f"seed=77 -> {s_a:.4f} (same for both instances)"
    return _check

def check_grader_reaches_success(task_id: str, fix_sqls: list, threshold: float):
    def _check():
        env = DataQualityEnv()
        obs = env.reset(task_id, seed=42)
        init_score = obs.quality_report.overall_score
        done = False
        for sql in fix_sqls:
            if done:
                break
            obs, _, done, _ = env.step(DataQualityAction(sql=sql))
        final = obs.quality_report.overall_score
        assert final >= threshold, f"Expert fixes did not reach threshold {threshold}: {final}"
        return f"{init_score:.4f} -> {final:.4f} (threshold={threshold})"
    return _check

for tid in ["null_patrol", "duplicate_destroyer", "constraint_cascade"]:
    check(f"grader score in [0,1] ({tid})", check_grader_scores_in_range(tid))
    check(f"grader deterministic ({tid})", check_grader_deterministic(tid))

check("T1 expert fixes reach threshold", check_grader_reaches_success(
    "null_patrol", [
        "UPDATE customers SET email = 'unknown@example.com' WHERE email IS NULL",
        "UPDATE customers SET phone = '000-000-0000' WHERE phone IS NULL",
    ], 0.95
))

check("T2 expert fixes reach threshold", check_grader_reaches_success(
    "duplicate_destroyer", [
        "DELETE FROM orders WHERE row_id NOT IN (SELECT MIN(row_id) FROM orders GROUP BY order_id)"
    ], 0.95
))

check("T3 expert fixes reach threshold", check_grader_reaches_success(
    "constraint_cascade", [
        "UPDATE products SET price = REPLACE(price, '$', '') WHERE price LIKE '$%'",
        "DELETE FROM inventory WHERE product_id NOT IN (SELECT product_id FROM products)",
        "UPDATE inventory SET quantity = ABS(quantity) WHERE quantity < 0",
    ], 0.90
))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reward function
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Validating reward function ...")

from reward import compute_reward

def check_reward_positive_for_improvement():
    r = compute_reward(0.5, 0.8, "UPDATE t SET x=1 WHERE x IS NULL", "success", 2, 15)
    assert r > 0, f"Expected positive reward, got {r}"
    return f"improvement reward={r:+.4f}"

def check_reward_negative_for_error():
    r = compute_reward(0.5, 0.5, "BAD SQL", "error: syntax error", 2, 15)
    assert r < 0, f"Expected negative reward for error, got {r}"
    return f"error penalty={r:+.4f}"

def check_reward_negative_for_drop():
    r = compute_reward(0.5, 0.5, "DROP TABLE customers", "success", 2, 15)
    assert r < 0, f"Expected negative reward for DROP, got {r}"
    return f"DROP penalty={r:+.4f}"

def check_reward_dense_signal():
    """Reachable partial improvement must produce non-zero reward."""
    r = compute_reward(0.5, 0.55, "UPDATE t SET x=1", "success", 3, 20)
    assert r != 0, "Reward for a 0.05 improvement should be non-zero"
    return f"partial improvement reward={r:+.4f}"

check("reward > 0 for improvement", check_reward_positive_for_improvement)
check("reward < 0 for SQL error", check_reward_negative_for_error)
check("reward < 0 for DROP TABLE", check_reward_negative_for_drop)
check("dense signal for partial improvement", check_reward_dense_signal)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Episode termination
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Validating episode termination ...")

def check_episode_done_on_success():
    env = DataQualityEnv()
    env.reset("null_patrol", seed=42)
    _, _, done, info = env.step(DataQualityAction(
        sql="UPDATE customers SET email = 'unknown@example.com' WHERE email IS NULL"
    ))
    obs, _, done, info = env.step(DataQualityAction(
        sql="UPDATE customers SET phone = '000-000-0000' WHERE phone IS NULL"
    ))
    assert done, f"Episode should be done after reaching success threshold"
    assert info.get("success") is True, "info['success'] should be True"
    return f"done={done}, success={info.get('success')}"

def check_error_sql_does_not_crash():
    env = DataQualityEnv()
    env.reset("null_patrol", seed=42)
    obs, reward, done, info = env.step(DataQualityAction(sql="THIS IS NOT VALID SQL!!!"))
    assert obs.last_action_result.startswith("error"), \
        f"Expected error result, got: {obs.last_action_result}"
    assert not done, "Invalid SQL should not end the episode"
    return f"error caught: '{obs.last_action_result[:40]}'"

check("episode done when success threshold reached", check_episode_done_on_success)
check("invalid SQL returns error observation (no crash)", check_error_sql_does_not_crash)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Required files
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Checking required files ...")

REQUIRED_FILES = [
    "app.py",
    "environment.py",
    "tasks.py",
    "data_generator.py",
    "reward.py",
    "inference.py",
    "openenv.yaml",
    "Dockerfile",
    "requirements.txt",
    "README.md",
]

def check_file_exists(fname: str):
    def _check():
        assert os.path.exists(fname), f"Missing required file: {fname}"
        size = os.path.getsize(fname)
        return f"{size} bytes"
    return _check

for fname in REQUIRED_FILES:
    check(f"file exists: {fname}", check_file_exists(fname))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
passed = sum(1 for ok, _, _ in results if ok)
failed = sum(1 for ok, _, _ in results if not ok)
total = len(results)

print("\n" + "=" * 60)
print(f"VALIDATION SUMMARY: {passed}/{total} checks passed")
print("=" * 60)

if failed > 0:
    print(f"\nFAILED CHECKS ({failed}):")
    for ok, name, msg in results:
        if not ok:
            print(f"  {FAIL} {name}: {msg}")
    print("\nFix all failures before submitting.")
    sys.exit(1)
else:
    print("\nAll checks passed! Ready to submit.")
    print("\nPre-submission checklist:")
    print("  [x] openenv.yaml valid")
    print("  [x] 3+ tasks with graders")
    print("  [x] Graders deterministic and in [0,1]")
    print("  [x] reset() / step() / state() work")
    print("  [x] Reward function provides useful signal")
    print("  [x] Episodes terminate correctly")
    print("  [x] All required files present")
    print("\nNext steps:")
    print("  1. docker build -t sql-data-quality-agent .")
    print("  2. docker run -p 7860:7860 sql-data-quality-agent")
    print("  3. Push to Hugging Face Spaces")
    sys.exit(0)
