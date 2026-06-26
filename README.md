# ConcurRL-vLLM

Advanced concurrency and latency benchmarking for agentic RL backends on vLLM.

## What This Project Does

Reinforcement learning (RL) pipelines that use LLMs as policy or reward models need to run hundreds to thousands of environment rollouts in parallel. The central question this project answers:

> **How does CPU-side scheduling, thread contention, and OS-level context switching degrade latency as concurrency scales from 32 to 1024 parallel rollouts on a vLLM backend?**

We isolate the CPU scheduling bottleneck from GPU compute by decomposing each request into 5 timed segments:

| Metric | What It Captures |
|---|---|
| Client Serialization | Time to serialize the JSON payload on the client side |
| Server HTTP Overhead | Scheduling, queue wait, and KV cache lookup before GPU work starts |
| GPU Prefill (TTFT) | Time from POST to first content token — includes overhead + prefill attention |
| GPU Decode | Autoregressive token generation interval (first → last token) |
| Response Parsing | Client-side SSE chunk parsing after the last token arrives |

By sweeping concurrency across `[32, 128, 256, 512, 1024]` and tracking these metrics, we can pinpoint exactly where the system transitions from nominal execution to queue contention to context-switch thrashing — and use that data to design targeted mitigations in Phase 3.

## Project Phases

### Phase 1: Concurrency Scaling Validation (Stress Testing)
- Set up mock environments and timing hooks to validate instrumentation
- Deploy vLLM with dual-GPU tensor parallelism
- Run async concurrency sweeps from 32 to 1024 parallel requests
- Collect per-request latency decomposition and OS-level telemetry

### Phase 2: Full RL Loop Integration
- Bridge the high-concurrency generation phase with centralized GPU/CPU training
- Measure overhead during the generation ↔ training phase transitions

### Phase 3: Analysis & Solution Design
- Parse execution timelines to find exact CPU stall and thrash points
- Prototype mitigations: scheduling-aware batching, dynamic thread pooling, async rollout/train overlaps

## Hardware Requirements

- **GPUs**: 2x NVIDIA A800-80GB (or equivalent with 80GB+ VRAM each)
- **VRAM Usage**: ~64GB for FP16 model weights (Qwen3-30B-A3B with TP=2), ~96GB free for KV cache
- **CPU**: Multi-core Xeon or equivalent (contention on this is what we're measuring)
- **RAM**: 128GB+ recommended

## Setup

```bash
# Clone and enter project
cd ConcurRL-vLLM-CPUbase

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `vllm>=0.4.0` | LLM serving engine |
| `aiohttp` | Async HTTP client for concurrency driver |
| `fastapi` + `uvicorn` | Mock server for compile check |
| `pydantic` | Data validation |
| `numpy` | Numerical operations |
| `psutil` | OS-level telemetry |

## Execution Order

### Step 1: Compile Check (no model download needed)

Validates that all 5 timing hooks work correctly using a lightweight mock server.

```bash
python script/01_compile_check.py
python script/01_compile_check.py --concurrency 8 --input-tokens 500
```

Output: `result/01_compile_check.json`

### Step 2: Download Model

Retrieve the model weights once (skip if already cached locally):

```bash
# The scripts default to "Qwen/Qwen3-30B-A3B"
# vLLM will auto-download from HuggingFace on first launch,
# or you can pre-download:
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ./models/Qwen3-30B-A3B
```

### Step 3: Launch vLLM Server

Start the production vLLM backend across both GPUs:

```bash
python script/02_launch_vllm.py
python script/02_launch_vllm.py --model ./models/Qwen3-30B-A3B --port 8000
```

Key flags (hardcoded defaults match Phase1Plan.md):
- `--tensor-parallel-size 2` — uses both CUDA 0 and CUDA 1
- `--max-model-len 4096` — context window cap
- `--enable-chunked-prefill` — shields RL loop from scheduling thrashing at 512+ concurrency
- `--gpu-memory-utilization 0.85` — leaves headroom for KV cache

### Step 4: Concurrency Stress Test

Run the async profiler across all concurrency tiers:

```bash
python script/03_concurrency_driver.py --url http://localhost:8000
python script/03_concurrency_driver.py --url http://localhost:8000 --scenarios 32 128 256
python script/03_concurrency_driver.py --url http://localhost:8000 --num-batches 5 --warmup-batches 2
```

Default sweep: `[32, 128, 256, 512, 1024]` concurrent requests with 32k input / 64 output tokens.

Output: `result/03_concurrency_driver.json`

### Step 5: Compile Metrics & Generate Report

Aggregate raw traces into percentile statistics and a Markdown summary:

```bash
python script/04_metrics_compiler.py
```

Output:
- `result/04_metrics_compiler.json` — structured analytics per concurrency tier
- `PHASE_1_SUMMARY.md` — scannable benchmark table with P50/P95/P99 breakdowns

## File Structure

```text
ConcurRL-vLLM-CPUbase/
├── .gitignore
├── README.md
├── PHASE_1_SUMMARY.md              # Generated benchmark report
├── Phase1PLAN.md                   # Detailed Phase 1 blueprint
├── PLAN.md                         # Full project roadmap (3 phases)
├── requirements.txt
│
├── script/
│   ├── __init__.py
│   ├── 01_compile_check.py         # Mock server + client timing validation
│   ├── 02_launch_vllm.py           # Production vLLM subprocess launcher (TP=2)
│   ├── 03_concurrency_driver.py    # Async burst profiler (32 → 1024)
│   └── 04_metrics_compiler.py      # JSON aggregator + Markdown visualizer
│
├── result/
│   ├── 01_compile_check.json       # Timing validation results
│   ├── 03_concurrency_driver.json  # Raw per-request traces
│   └── 04_metrics_compiler.json    # Aggregated P50/P95/P99 analytics
│
└── reference/
    └── 04_server_decomposition.py  # Reference coding style (HBM prefix cache benchmark)
```

## CLI Reference

All scripts support `--help` for full flag documentation.

| Script | Key Flags | Default |
|---|---|---|
| `01_compile_check.py` | `--concurrency`, `--input-tokens`, `--output-tokens`, `--port` | 4 concurrent, 100 input tokens |
| `02_launch_vllm.py` | `--model`, `--port`, `--tensor-parallel-size`, `--max-model-len`, `--wait-only` | Qwen3-30B-A3B, port 8000, TP=2 |
| `03_concurrency_driver.py` | `--url`, `--scenarios`, `--num-batches`, `--input-tokens`, `--output-tokens` | localhost:8000, [32..1024], 3 batches |
| `04_metrics_compiler.py` | `--input`, `--output-json`, `--output-md` | Reads from `result/03_*.json` |
