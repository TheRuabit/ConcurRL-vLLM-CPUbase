#!/usr/bin/env python3
"""
Phase 2: GRPO RL Loop via veRL
================================
Wraps veRL's `verl.trainer.main_ppo` with CLI args and hardware-appropriate
defaults for 2x A800-80GB. After training, extracts veRL's built-in per-stage
timing data and writes it to `result/05_grpo_train.json`.

veRL instruments every training stage with `marked_timer`, producing
`timing_s/{stage}` metrics. This script captures those metrics and reformats
them for Phase 2 analysis.

Prerequisites:
  1. veRL installed (pip install verl or from source)
  2. DAPO-Math-17k downloaded to data/dapo-math-17k.parquet
  3. Model weights available (Qwen/Qwen3-30B-A3B)

Usage:
    python script/05_grpo_train.py
    python script/05_grpo_train.py --model ./models/Qwen3-30B-A3B --num-epochs 3
    python script/05_grpo_train.py --train-batch-size 16 --rollout-n 4

Output:
    result/05_grpo_train.json — per-step timing breakdown
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Phase 2: GRPO RL loop via veRL with timing instrumentation"
)
parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                    help="Model name or path")
parser.add_argument("--data-path", default=None,
                    help="Path to DAPO-Math-17k parquet (default: data/dapo-math-17k.parquet)")
parser.add_argument("--val-path", default=None,
                    help="Path to validation parquet (default: data/aime-2024.parquet)")
parser.add_argument("--output", default=None,
                    help="Output JSON path")
parser.add_argument("--rollout-n", type=int, default=8,
                    help="Number of responses per prompt (GRPO group size)")
parser.add_argument("--train-batch-size", type=int, default=32,
                    help="Prompts per training step")
parser.add_argument("--ppo-mini-batch-size", type=int, default=None,
                    help="PPO mini-batch size (default: same as train-batch-size)")
parser.add_argument("--max-prompt-length", type=int, default=2048,
                    help="Max prompt token length")
parser.add_argument("--max-response-length", type=int, default=4096,
                    help="Max response token length")
parser.add_argument("--actor-lr", type=float, default=1e-6,
                    help="Actor learning rate")
parser.add_argument("--num-epochs", type=int, default=3,
                    help="Number of training epochs")
parser.add_argument("--rollout-tp", type=int, default=2,
                    help="Tensor parallel size for vLLM rollout")
parser.add_argument("--rollout-gpu-mem-util", type=float, default=0.65,
                    help="GPU memory utilization for vLLM rollout")
parser.add_argument("--kl-loss-coef", type=float, default=0.001,
                    help="KL loss coefficient")
parser.add_argument("--reward-func", default=None,
                    help="Path to custom reward function (default: script/06_math_reward.py)")
parser.add_argument("--save-freq", type=int, default=50,
                    help="Checkpoint save frequency (steps)")
parser.add_argument("--test-freq", type=int, default=5,
                    help="Validation frequency (steps)")
parser.add_argument("--log-file", default=None,
                    help="Capture veRL stdout/stderr to this file")
parser.add_argument("--extra-overrides", nargs="*", default=[],
                    help="Additional Hydra overrides (key=value)")
args = parser.parse_args()

if args.output:
    out_path = Path(args.output)
else:
    out_path = Path(__file__).resolve().parents[1] / "result" / "05_grpo_train.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

project_dir = Path(__file__).resolve().parents[1]

# Resolve data paths
if args.data_path is None:
    args.data_path = str(project_dir / "data" / "dapo-math-17k.parquet")
if args.val_path is None:
    args.val_path = str(project_dir / "data" / "aime-2024.parquet")
if args.reward_func is None:
    args.reward_func = str(project_dir / "script" / "06_math_reward.py")

# Default ppo_mini_batch_size = train_batch_size
if args.ppo_mini_batch_size is None:
    args.ppo_mini_batch_size = args.train_batch_size


def build_hydra_overrides() -> list[str]:
    """Convert CLI args to veRL Hydra config overrides."""
    overrides = [
        # Data
        f"data.train_files=['{args.data_path}']",
        f"data.val_files=['{args.val_path}']",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",

        # Model
        f"actor_rollout_ref.model.path={args.model}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",

        # Algorithm
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",

        # Actor (FSDP2 with offloading for 2-GPU setup)
        "actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.optim.lr={args.actor_lr}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={args.kl_loss_coef}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",

        # Rollout (vLLM)
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rollout_gpu_mem_util}",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768",

        # Reference policy
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",

        # Reward
        f"reward.custom_reward_function.path={args.reward_func}",
        "reward.custom_reward_function.name=compute_score",

        # Trainer
        "trainer.balance_batch=True",
        "trainer.logger=['console']",
        "trainer.project_name=concurrl_phase2",
        "trainer.experiment_name=grpo_qwen3_30b_a3b",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.test_freq}",
        f"trainer.total_epochs={args.num_epochs}",
    ]

    # Extra overrides
    overrides.extend(args.extra_overrides)

    return overrides


def parse_timing_from_log(log_text: str) -> dict:
    """Extract timing metrics from veRL's console output.

    veRL logs timing as:
      timing_s/gen: 12.345
      timing_s/reward: 0.123
      timing_s/old_log_prob: 2.345
      ...
    """
    timing_pattern = re.compile(r"timing_s/(\w+):\s+([\d.]+)")
    steps = []
    current_step = {}

    for line in log_text.split("\n"):
        # Check for step boundary
        if "step" in line.lower() and ("epoch" in line.lower() or "global_step" in line.lower()):
            if current_step:
                steps.append(current_step)
                current_step = {}

        match = timing_pattern.search(line)
        if match:
            stage_name = match.group(1)
            timing_s = float(match.group(2))
            current_step[f"t_{stage_name}_s"] = timing_s

    if current_step:
        steps.append(current_step)

    return steps


def main():
    print(f"[05_grpo_train] Phase 2: GRPO RL Loop via veRL")
    print(f"[05_grpo_train] Model:      {args.model}")
    print(f"[05_grpo_train] Data:       {args.data_path}")
    print(f"[05_grpo_train] Rollout N:  {args.rollout_n}")
    print(f"[05_grpo_train] Batch size: {args.train_batch_size}")
    print(f"[05_grpo_train] Epochs:     {args.num_epochs}")
    print(f"[05_grpo_train] Output:     {out_path}")
    print()

    # Build command
    overrides = build_hydra_overrides()
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo"] + overrides

    print(f"[05_grpo_train] Launching veRL GRPO trainer...")
    print(f"[05_grpo_train] Command: {' '.join(cmd[:6])} ... ({len(overrides)} overrides)")
    print()

    # Prepare log file
    log_handle = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "w")
        print(f"[05_grpo_train] Logging to: {log_path}")

    t_start = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if not log_handle else log_handle,
            stderr=subprocess.STDOUT if not log_handle else None,
            text=True,
            cwd=str(project_dir),
        )

        t_elapsed = time.perf_counter() - t_start

        if result.returncode != 0:
            print(f"[05_grpo_train] ERROR: veRL exited with code {result.returncode}")
            if not log_handle and result.stdout:
                # Print last 30 lines of output
                lines = result.stdout.strip().split("\n")
                for line in lines[-30:]:
                    print(f"  {line}")
            sys.exit(1)

        print(f"[05_grpo_train] veRL completed in {t_elapsed:.1f}s")

    finally:
        if log_handle:
            log_handle.close()

    # Parse timing from log file
    log_text = ""
    if args.log_file:
        log_text = Path(args.log_file).read_text(encoding="utf-8", errors="replace")
    elif result.stdout:
        log_text = result.stdout

    step_timings = parse_timing_from_log(log_text)

    # Build output
    output = {
        "benchmark": "phase2_grpo_rl_loop",
        "model": args.model,
        "data_path": args.data_path,
        "rollout_n": args.rollout_n,
        "train_batch_size": args.train_batch_size,
        "ppo_mini_batch_size": args.ppo_mini_batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "num_epochs": args.num_epochs,
        "rollout_tp": args.rollout_tp,
        "total_elapsed_s": round(t_elapsed, 2),
        "num_steps": len(step_timings),
        "step_timings": step_timings,
        "verl_overrides": overrides,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[05_grpo_train] Results saved to {out_path}")

    # Print timing summary
    if step_timings:
        print(f"\n{'='*80}")
        print(f"GRPO TRAINING — Per-Step Timing Summary")
        print(f"{'='*80}")
        for i, step in enumerate(step_timings):
            parts = []
            for k, v in sorted(step.items()):
                parts.append(f"{k}={v:.2f}s")
            print(f"  Step {i:3d}: {', '.join(parts)}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
