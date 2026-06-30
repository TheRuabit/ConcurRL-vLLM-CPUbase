# ConcurRL-vLLM

Concurrency and latency benchmarking for agentic RL backends on vLLM.

## Overview

RL pipelines that use LLMs as policy or reward models need hundreds to thousands of parallel rollouts. This project measures how CPU-side scheduling, thread contention, and OS-level context switching degrade latency as concurrency scales from 32 to 1024 on a vLLM backend.

Each request is decomposed into non-overlapping timed segments:

| Metric               | What It Captures                                                             |
| -------------------- | ---------------------------------------------------------------------------- |
| Client Serialization | JSON payload serialization on the client                                     |
| HTTP Overhead        | POST → first SSE byte (scheduling, queue wait, KV cache lookup on localhost) |
| TTFT                 | POST → first content token (HTTP overhead + GPU prefill)                     |
| GPU Prefill          | Pure GPU attention over input tokens (TTFT − HTTP overhead)                  |
| GPU Decode           | Autoregressive token generation (first → last token)                         |
| Response Parsing     | Client-side SSE stream consumption (first byte → end of stream)              |

CPU time = Serialization + HTTP Overhead. GPU time = Prefill + Decode. These are non-overlapping and sum to E2E.

## Project Phases

**Phase 1 — Concurrency Scaling Validation**: Async concurrency sweeps [32, 128, 256, 512, 1024] with per-request latency decomposition.

**Phase 2 — Full RL Loop Integration**: GRPO training via veRL, measuring per-step timing breakdown (rollout → reward → advantage → train → weight sync) to validate Phase 1's CPU bottleneck finding inside a real RL loop.

**Phase 3 — Analysis & Solution Design**: Bottleneck profiling and mitigation prototyping. (TODO)

## Hardware Requirements

- **GPUs**: 2x NVIDIA A800-80GB (or equivalent, 80GB+ VRAM each)
- **VRAM**: ~64GB for FP16 weights (Qwen3-30B-A3B, TP=2), remainder for KV cache
- **CPU**: Multi-core Xeon or equivalent
- **RAM**: 128GB+ recommended

## Setup

```bash
cd ConcurRL-vLLM-CPUbase
python -m venv venv
source venv/bin/activate        # Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

| Package               | Purpose                                  |
| --------------------- | ---------------------------------------- |
| `vllm>=0.4.0`         | LLM serving engine                       |
| `aiohttp`             | Async HTTP client for concurrency driver |
| `fastapi` + `uvicorn` | Mock server for compile check            |
| `pydantic`            | Data validation                          |
| `numpy`               | Numerical operations                     |
| `psutil`              | OS-level telemetry                       |

## Quick Start (run_all.sh)

The easiest way to run the full pipeline:

```bash
# Phase 1: compile check + vLLM concurrency sweep
bash run_all.sh --phase 1

# Phase 2: GRPO RL loop via veRL (downloads DAPO-Math-17k automatically)
bash run_all.sh --phase 2

# Keep vLLM server alive after completion (for further experiments)
bash run_all.sh --phase 1 --keep-vllm

# All phases
bash run_all.sh --phase all

# Custom model and output tokens
bash run_all.sh --phase 1 --model ./models/Qwen3-30B-A3B --max_output_token 128
```

`run_all.sh` launches vLLM automatically, runs all steps, and cleans up the server on exit (unless `--keep-vllm`).

## Manual Execution

### Step 1: Compile Check

Validates timing hooks using a lightweight mock server (no model download needed):

```bash
python script/01_compile_check.py
```

Output: `result/01_compile_check.json`

### Step 2: Download Model

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ./models/Qwen3-30B-A3B
```

vLLM will also auto-download on first launch.

### Step 3: Launch vLLM Server

```bash
python script/02_launch_vllm.py
python script/02_launch_vllm.py --model ./models/Qwen3-30B-A3B --port 8000
```

Key flags: `--tensor-parallel-size 2`, `--max-model-len 51200`, `--enable-chunked-prefill`, `--gpu-memory-utilization 0.85`. Use `--detach --pid-file <path>` to launch in background (exits after health check, writes PID to file).

### Step 4: Concurrency Stress Test

```bash
python script/03_concurrency_driver.py --url http://localhost:8000
python script/03_concurrency_driver.py --url http://localhost:8000 --scenarios 32 128 256
```

Default sweep: [32, 128, 256, 512, 1024] concurrent requests, 32k input / 64 output tokens, 3 batches.

Output: `result/03_concurrency_driver.json`

### Step 5: Compile Metrics & Generate Report

```bash
python script/04_metrics_compiler.py
```

Output:

- `result/04_metrics_compiler.json` — structured analytics per tier
- `PHASE_1_SUMMARY.md` — benchmark tables with P50/P95/P99 breakdowns

## File Structure

```text
ConcurRL-vLLM-CPUbase/
├── run_all.sh                      # One-command pipeline orchestration
├── README.md
├── PHASE_1_SUMMARY.md              # Phase 1 benchmark report
├── PHASE_2_SUMMARY.md              # Phase 2 GRPO timing report
├── Phase1PLAN.md / Phase2PLAN.md   # Phase blueprints
├── PLAN.md                         # Full project roadmap
├── requirements.txt
│
├── script/
│   ├── 01_compile_check.py         # Mock server + client timing validation
│   ├── 02_launch_vllm.py           # vLLM server launcher (supports --detach)
│   ├── 03_concurrency_driver.py    # Async burst profiler (32 → 1024)
│   ├── 04_metrics_compiler.py      # JSON aggregator + Markdown report
│   ├── 05_grpo_train.py            # Phase 2: GRPO training via veRL
│   ├── 06_math_reward.py           # Phase 2: math reward function with timing
│   └── 07_phase2_metrics.py        # Phase 2: timing compiler
│
├── data/                           # Datasets (auto-downloaded)
│   ├── dapo-math-17k.parquet
│   └── aime-2024.parquet
│
├── result/                         # All output JSONs and PID files
│
└── reference/
    └── 04_server_decomposition.py  # Reference decomposition (HBM prefix cache)
```

## CLI Reference

All scripts support `--help`.

| Script                     | Key Flags                                                                                                 | Default                               |
| -------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `01_compile_check.py`      | `--concurrency`, `--input-tokens`, `--output-tokens`                                                      | 4 concurrent, 100 input tokens        |
| `02_launch_vllm.py`        | `--model`, `--port`, `--tensor-parallel-size`, `--max-model-len`, `--detach`, `--pid-file`, `--wait-only` | Qwen3-30B-A3B, port 8000, TP=2        |
| `03_concurrency_driver.py` | `--url`, `--scenarios`, `--num-batches`, `--input-tokens`, `--output-tokens`                              | localhost:8000, [32..1024], 3 batches |
| `04_metrics_compiler.py`   | `--input`, `--output-json`, `--output-md`                                                                 | Reads from `result/03_*.json`         |
| `05_grpo_train.py`         | `--model`, `--rollout-n`, `--train-batch-size`, `--num-epochs`, `--reward-func`                           | rollout_n=8, batch=32, epochs=3       |
| `07_phase2_metrics.py`     | `--input`, `--output-json`, `--output-md`                                                                 | Reads from `result/05_*.json`         |
| `run_all.sh`               | `--phase`, `--model`, `--url`, `--max_output_token`, `--keep-vllm`                                        | phase=all, Qwen3-30B-A3B              |
