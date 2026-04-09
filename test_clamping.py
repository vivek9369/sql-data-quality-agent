import json
import logging
from tasks import clamp_score, TASK_REGISTRY

def test_inference_clamping():
    def clamp_val(v: float, low: float = 0.01, high: float = 0.99) -> float:
        result = max(low, min(high, v))
        if result <= 0.0:
            result = low
        if result >= 1.0:
            result = high
        return result

    for v in [0.0, 0.001, 0.005, 0.009, 0.01, 1.0, 0.99, 0.995, 0.999, 0.9999]:
        clamped = clamp_val(v)
        fmt = f"{clamped:.2f}"
        print(f"v={v} -> clamped={clamped} -> fmt={fmt}")

test_inference_clamping()
