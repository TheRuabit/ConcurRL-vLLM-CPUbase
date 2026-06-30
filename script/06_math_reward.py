#!/usr/bin/env python3
"""
Math Reward Function with Timing Instrumentation
=================================================
A veRL-compatible reward function for GRPO training on math tasks.
Wraps answer extraction and comparison with `time.perf_counter()` to
measure the CPU-side reward computation overhead.

Compatible with DAPO-Math-17k (data_source="math_dapo") and GSM8K.

Usage (via veRL config):
    reward.custom_reward_function.path=script/06_math_reward.py
    reward.custom_reward_function.name=compute_score

Standalone test:
    python script/06_math_reward.py
"""

import json
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Timing accumulator (module-level, read by 07_phase2_metrics.py)
# ---------------------------------------------------------------------------
_TIMINGS: list[dict] = []


def get_reward_timings() -> list[dict]:
    """Return accumulated reward timing records."""
    return list(_TIMINGS)


def reset_reward_timings():
    """Clear accumulated timings (useful between runs)."""
    _TIMINGS.clear()


def save_reward_timings(path: str = None):
    """Write timings to JSON file."""
    if path is None:
        path = str(Path(__file__).resolve().parents[1] / "result" / "06_reward_timings.json")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_TIMINGS, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Answer extraction (DAPO-Math style)
# ---------------------------------------------------------------------------
def extract_boxed(text: str) -> str:
    """Extract the last \\boxed{...} answer from model output."""
    # Find all \boxed{...} with brace matching
    results = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        # Skip past \boxed{
        j = idx + len(r"\boxed{")
        depth = 1
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[idx + len(r"\boxed{"):j - 1])
        i = j
    return results[-1].strip() if results else ""


def normalize_answer(answer: str) -> str:
    """Normalize math answer for comparison (Minerva-style)."""
    answer = answer.strip()
    # Remove \text{}, \mathrm{}, etc.
    answer = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{([^}]*)\}", r"\1", answer)
    # Remove \left, \right
    answer = re.sub(r"\\(?:left|right)", "", answer)
    # Remove spaces
    answer = re.sub(r"\s+", "", answer)
    # Remove trailing period
    answer = answer.rstrip(".")
    # Normalize fractions: \frac{a}{b} -> a/b
    answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", answer)
    return answer.lower()


# ---------------------------------------------------------------------------
# Main reward function (veRL interface)
# ---------------------------------------------------------------------------
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> float:
    """
    veRL-compatible reward function.

    Args:
        data_source: Dataset identifier (e.g., "math_dapo", "openai/gsm8k")
        solution_str: Model's generated response text
        ground_truth: Expected answer (extracted during data preprocessing)
        extra_info: Optional metadata dict

    Returns:
        1.0 for correct answer, 0.0 for incorrect
    """
    t0 = time.perf_counter()

    # Extract answer from model output
    t_extract_start = time.perf_counter()
    extracted = extract_boxed(solution_str)
    t_extract_end = time.perf_counter()

    # Normalize and compare
    t_compare_start = time.perf_counter()
    if extracted:
        norm_extracted = normalize_answer(extracted)
        norm_gt = normalize_answer(ground_truth)
        correct = norm_extracted == norm_gt
    else:
        correct = False
    t_compare_end = time.perf_counter()

    reward = 1.0 if correct else 0.0

    t_total = time.perf_counter()

    _TIMINGS.append({
        "data_source": data_source,
        "t_extract_ms": round((t_extract_end - t_extract_start) * 1000, 4),
        "t_compare_ms": round((t_compare_end - t_compare_start) * 1000, 4),
        "t_total_ms": round((t_total - t0) * 1000, 4),
        "correct": correct,
        "extracted": extracted[:100] if extracted else "",
    })

    return reward


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[06_reward] Testing math reward function...")

    test_cases = [
        ("The answer is \\boxed{42}.", "42", True),
        ("We get \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}", True),
        ("\\boxed{3.14}", "3.14", True),
        ("No boxed answer here.", "42", False),
        ("\\boxed{100}", "99", False),
    ]

    for sol, gt, expected in test_cases:
        result = compute_score("test", sol, gt)
        status = "OK" if (result > 0) == expected else "FAIL"
        print(f"  [{status}] sol={sol[:30]:30s} gt={gt:6s} → {result}")

    timings = get_reward_timings()
    print(f"\n  Timings ({len(timings)} calls):")
    avg_ms = sum(t["t_total_ms"] for t in timings) / len(timings)
    print(f"    Avg reward compute: {avg_ms:.3f}ms")

    save_reward_timings()
    print(f"  Saved to result/06_reward_timings.json")
