"""Full verification test for all 3 tasks (OpenEnv spec compliance check)."""
import sys
# Ensure UTF-8 output on all platforms
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from environment import DataQualityEnv, DataQualityAction
from reward import compute_reward

print("=" * 50)
print("SQL DATA QUALITY AGENT - FULL VERIFICATION")
print("=" * 50)

# --- Task 1: Null Patrol (Easy) ---
env1 = DataQualityEnv()
obs = env1.reset("null_patrol", seed=42)
t1_init = obs.quality_report.overall_score
print(f"\n[T1] Null Patrol (Easy)")
print(f"  Initial score: {t1_init:.4f}")
assert 0.0 <= t1_init <= 1.0, "Score must be in [0,1]"
obs, r, done, _ = env1.step(DataQualityAction(sql="UPDATE customers SET email = 'unknown@example.com' WHERE email IS NULL"))
print(f"  After email fix: score={obs.quality_report.overall_score:.4f}, reward={r:+.4f}")
obs, r, done, _ = env1.step(DataQualityAction(sql="UPDATE customers SET phone = '000-000-0000' WHERE phone IS NULL"))
print(f"  After phone fix: score={obs.quality_report.overall_score:.4f}, done={done}, reward={r:+.4f}")
assert obs.quality_report.overall_score >= 0.95, f"T1 FAIL: {obs.quality_report.overall_score}"
print(f"  [T1] PASSED ({t1_init:.4f} -> {obs.quality_report.overall_score:.4f})")

# --- Task 2: Duplicate Destroyer (Medium) ---
env2 = DataQualityEnv()
obs = env2.reset("duplicate_destroyer", seed=42)
t2_init = obs.quality_report.overall_score
print(f"\n[T2] Duplicate Destroyer (Medium)")
print(f"  Initial score: {t2_init:.4f}")
assert 0.0 <= t2_init <= 1.0, "Score must be in [0,1]"
obs, r, done, _ = env2.step(DataQualityAction(
    sql="DELETE FROM orders WHERE row_id NOT IN (SELECT MIN(row_id) FROM orders GROUP BY order_id)"
))
print(f"  After dedup: score={obs.quality_report.overall_score:.4f}, done={done}, reward={r:+.4f}")
assert obs.quality_report.overall_score >= 0.95, f"T2 FAIL: {obs.quality_report.overall_score}"
print(f"  [T2] PASSED ({t2_init:.4f} -> {obs.quality_report.overall_score:.4f})")

# --- Task 3: Constraint Cascade (Hard) ---
env3 = DataQualityEnv()
obs = env3.reset("constraint_cascade", seed=42)
t3_init = obs.quality_report.overall_score
print(f"\n[T3] Constraint Cascade (Hard)")
print(f"  Initial score: {t3_init:.4f}")
assert 0.0 <= t3_init <= 1.0, "Score must be in [0,1]"

fixes = [
    "UPDATE products SET price = REPLACE(price, '$', '') WHERE price LIKE '$%'",
    "DELETE FROM inventory WHERE product_id NOT IN (SELECT product_id FROM products)",
    "UPDATE inventory SET quantity = ABS(quantity) WHERE quantity < 0",
]
done = False
for sql in fixes:
    if done:
        break
    obs, r, done, _ = env3.step(DataQualityAction(sql=sql))
    print(f"  score={obs.quality_report.overall_score:.4f}, done={done}, reward={r:+.4f}")
assert obs.quality_report.overall_score >= 0.90, f"T3 FAIL: {obs.quality_report.overall_score}"
print(f"  [T3] PASSED ({t3_init:.4f} -> {obs.quality_report.overall_score:.4f})")

# --- Reward sanity: scores must be in [0,1] and reward signal must work ---
r_pos = compute_reward(0.5, 0.8, "UPDATE x SET y=1 WHERE y IS NULL", "success", 2, 15)
r_err = compute_reward(0.5, 0.5, "BAAD SQL", "error: syntax error", 2, 15)
r_drop = compute_reward(0.5, 0.5, "DROP TABLE customers", "success", 2, 15)
print(f"\n[Reward] positive delta -> {r_pos:+.4f} (expect >0)")
print(f"[Reward] error action   -> {r_err:+.4f} (expect <0)")
print(f"[Reward] DROP TABLE     -> {r_drop:+.4f} (expect <<0)")
assert r_pos > 0, f"Expected positive reward for improvement, got {r_pos}"
assert r_err < 0, f"Expected negative reward for error, got {r_err}"
assert r_drop < 0, f"Expected negative reward for DROP, got {r_drop}"
print("  [Reward] PASSED")

# --- Grader determinism: same seed must produce same initial score ---
env_a = DataQualityEnv()
env_b = DataQualityEnv()
obs_a = env_a.reset("null_patrol", seed=99)
obs_b = env_b.reset("null_patrol", seed=99)
assert obs_a.quality_report.overall_score == obs_b.quality_report.overall_score, \
    "Grader is not deterministic!"
print(f"\n[Determinism] seed=99 initial score (both instances): {obs_a.quality_report.overall_score:.4f} PASSED")

# --- Grader scores must be in valid range ---
for task_id in ["null_patrol", "duplicate_destroyer", "constraint_cascade"]:
    env_t = DataQualityEnv()
    obs_t = env_t.reset(task_id, seed=42)
    score = obs_t.quality_report.overall_score
    assert 0.0 <= score <= 1.0, f"Score out of range for {task_id}: {score}"
    print(f"[Grader range] {task_id}: {score:.4f} in [0,1] PASSED")

# --- State endpoint ---
st = env3.state()
assert st.task_id == "constraint_cascade"
assert st.step >= 1
assert st.episode_id is not None
print(f"\n[State] task={st.task_id}, step={st.step}, score={st.current_score:.4f}")
print("  [State] PASSED")

print("\n" + "=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
