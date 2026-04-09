import requests
import time
import sys

BASE_URL = "http://localhost:7860"

def test_api():
    tasks = requests.get(f"{BASE_URL}/tasks").json()["tasks"]
    for task in tasks:
        task_id = task["task_id"]
        print(f"Testing task: {task_id}")
        
        obs = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": 42}).json()
        overall_score = obs["quality_report"]["overall_score"]
        print(f"  Reset score: {overall_score}")
        if not (0.0 < overall_score < 1.0):
            print(f"  FAILED: Reset score {overall_score} out of range!")
            
        step_res = requests.post(f"{BASE_URL}/step", json={"sql": "SELECT 1;"}).json()
        step_score = step_res["observation"]["quality_report"]["overall_score"]
        step_reward = step_res["reward"]
        print(f"  Step score: {step_score}")
        print(f"  Step reward: {step_reward}")
        if not (0.0 < step_score < 1.0):
            print(f"  FAILED: Step score {step_score} out of range!")
        if not (0.0 < step_reward < 1.0):
            print(f"  FAILED: Step reward {step_reward} out of range!")

if __name__ == "__main__":
    test_api()
