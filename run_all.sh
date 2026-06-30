#!/bin/bash
# ==============================================================================
# Full Benchmark Orchestration — ConcurRL-vLLM
# ==============================================================================
# Runs the benchmark pipeline in recommended execution order:
#
#   Phase test — Compile check (mock server, no model needed)
#     01_compile_check.py
#
#   Phase 1 — Concurrency Scaling Validation
#     02_launch_vllm.py       (production vLLM server, TP=2)
#     03_concurrency_driver.py (async sweep 32 → 1024)
#     04_metrics_compiler.py   (JSON aggregator + Markdown report)
#
#   Phase 2 — Full RL Loop Integration (veRL GRPO)
#     05_grpo_train.py        (GRPO training via veRL)
#     06_math_reward.py        (math reward function)
#     07_phase2_metrics.py     (Phase 2 timing compiler)
#
#   Phase 3 — Analysis & Solution Design  [TODO]
#
# Usage:
#   bash run_all.sh --phase test                          # Compile check only
#   bash run_all.sh --phase 1                             # Full Phase 1 pipeline
#   bash run_all.sh --phase all                            # Everything
#   bash run_all.sh --phase 1 --model ./models/Qwen3-30B-A3B
#   bash run_all.sh --phase 1 --max_output_token 128
#   bash run_all.sh --phase 1 --url http://localhost:8000
#   bash run_all.sh --phase 1 --keep-vllm              # Keep server alive after run
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------------------------------------------------------------------
# Default settings
# -------------------------------------------------------------------
PHASE="all"
SERVER_URL="http://localhost:8000"
MODEL="Qwen/Qwen3-30B-A3B"
MAX_OUTPUT_TOKEN=64
PYTHON="python"
SKIP_VENV=false
KEEP_VLLM=false

# -------------------------------------------------------------------
# Parse arguments
# -------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)            PHASE="$2";            shift 2 ;;
        --url)              SERVER_URL="$2";        shift 2 ;;
        --model)            MODEL="$2";             shift 2 ;;
        --max_output_token) MAX_OUTPUT_TOKEN="$2";  shift 2 ;;
        --max-output-token) MAX_OUTPUT_TOKEN="$2";  shift 2 ;;
        --python)           PYTHON="$2";            shift 2 ;;
        --skip_venv)        SKIP_VENV=true;         shift  ;;
        --keep-vllm)        KEEP_VLLM=true;         shift  ;;
        -h|--help)
            echo "Usage: bash run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --phase <test|1|2|all>  Which phase to run (default: all)"
            echo "                          test  = 01_compile_check.py (mock server)"
            echo "                          1     = 02+03+04 (vLLM concurrency sweep)"
            echo "                          2     = 05+06+07 (GRPO RL loop via veRL)"
            echo "                          all   = all phases"
            echo "  --model <path>          Model name or path (default: Qwen/Qwen3-30B-A3B)"
            echo "  --max_output_token <N>  Max output tokens per request (default: 64)"
            echo "  --url <url>             vLLM server URL (default: http://localhost:8000)"
            echo "  --python <cmd>          Python interpreter (default: python)"
            echo "  --keep-vllm             Keep vLLM server running after pipeline completes"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run 'bash run_all.sh --help' for usage."
            exit 1
            ;;
    esac
done

# -------------------------------------------------------------------
# Activate venv if present
# -------------------------------------------------------------------
if ! $SKIP_VENV && [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate 2>/dev/null || true
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate 2>/dev/null || true
fi

# Ensure output directories exist
mkdir -p result

# -------------------------------------------------------------------
# Print config
# -------------------------------------------------------------------
echo "============================================"
echo " ConcurRL-vLLM Benchmark Pipeline"
echo "============================================"
echo " Phase:             $PHASE"
echo " Model:             $MODEL"
echo " Max Output Tokens: $MAX_OUTPUT_TOKEN"
echo " Server URL:        $SERVER_URL"
echo " Keep vLLM:         $KEEP_VLLM"
echo "============================================"
echo ""

# ===================================================================
# Phase test: Compile Check (mock server, no model needed)
# ===================================================================
run_phase_test() {
    echo "==========================================="
    echo " PHASE TEST: Compile Check (Mock Server)"
    echo "==========================================="

    echo ""
    echo "--- 01 Compile Check ---"
    $PYTHON script/01_compile_check.py \
        --output-tokens "$MAX_OUTPUT_TOKEN"
    echo "  OK: result/01_compile_check.json"

    echo ""
    echo "[Phase test] Complete."
}

# ===================================================================
# Phase 1: Concurrency Scaling Validation
# ===================================================================
VLLM_PID_FILE=""

run_phase_1() {
    echo "==========================================="
    echo " PHASE 1: Concurrency Scaling Validation"
    echo "==========================================="

    # Check if server is already running
    echo "[Phase1] Checking server at $SERVER_URL ..."
    if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo "[Phase1] Server already running, skipping launch."
    else
        echo ""
        echo "--- 02 Launch vLLM Server ---"
        echo "[Phase1] Starting vLLM server (this may take a few minutes)..."
        VLLM_PID_FILE="result/vllm_server.pid"
        $PYTHON script/02_launch_vllm.py \
            --model "$MODEL" \
            --port "${SERVER_URL##*:}" \
            --detach \
            --pid-file "$VLLM_PID_FILE"
    fi

    # Concurrency sweep
    echo ""
    echo "--- 03 Concurrency Driver ---"
    $PYTHON script/03_concurrency_driver.py \
        --url "$SERVER_URL" \
        --model "$MODEL" \
        --output-tokens "$MAX_OUTPUT_TOKEN" \
        --scenarios 32 128 256 512 1024 \
        --num-batches 3 \
        --warmup-batches 1
    echo "  OK: result/03_concurrency_driver.json"

    # Compile metrics
    echo ""
    echo "--- 04 Metrics Compiler ---"
    $PYTHON script/04_metrics_compiler.py
    echo "  OK: result/04_metrics_compiler.json"
    echo "  OK: PHASE_1_SUMMARY.md"

    echo ""
    echo "[Phase 1] Complete. Results in ./result/ and PHASE_1_SUMMARY.md"
}

# ===================================================================
# Phase 2: Full RL Loop Integration (veRL GRPO)
# ===================================================================
run_phase_2() {
    echo "==========================================="
    echo " PHASE 2: Full RL Loop Integration"
    echo "==========================================="

    # Download DAPO-Math-17k if not present
    if [ ! -f "data/dapo-math-17k.parquet" ]; then
        echo ""
        echo "--- Download DAPO-Math-17k ---"
        mkdir -p data
        echo "[Phase2] Downloading DAPO-Math-17k..."
        wget -q --show-progress -O data/dapo-math-17k.parquet \
            "https://hf-mirror.com//datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true" \
            || { echo "[Phase2] ERROR: Failed to download DAPO-Math-17k"; exit 1; }
        echo "[Phase2] OK: data/dapo-math-17k.parquet"
    else
        echo "[Phase2] DAPO-Math-17k already present."
    fi

    # Download AIME-2024 validation set if not present
    if [ ! -f "data/aime-2024.parquet" ]; then
        echo "[Phase2] Downloading AIME-2024..."
        wget -q --show-progress -O data/aime-2024.parquet \
            "https://hf-mirror.com//datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true" \
            || echo "[Phase2] WARNING: Failed to download AIME-2024 (validation will be skipped)"
        echo "[Phase2] OK: data/aime-2024.parquet"
    fi

    # GRPO training
    echo ""
    echo "--- 05 GRPO Train (veRL) ---"
    $PYTHON script/05_grpo_train.py \
        --model "$MODEL" \
        --rollout-n 8 \
        --train-batch-size 32 \
        --num-epochs 3 \
        --log-file "result/05_verl_training.log"
    echo "  OK: result/05_grpo_train.json"

    # Compile Phase 2 metrics
    echo ""
    echo "--- 07 Phase 2 Metrics ---"
    $PYTHON script/07_phase2_metrics.py
    echo "  OK: result/07_phase2_metrics.json"
    echo "  OK: PHASE_2_SUMMARY.md"

    echo ""
    echo "[Phase 2] Complete. Results in ./result/ and PHASE_2_SUMMARY.md"
}

# ===================================================================
# Phase 3: Analysis & Solution Design [TODO]
# ===================================================================
run_phase_3() {
    echo "==========================================="
    echo " PHASE 3: Analysis & Solution Design"
    echo "==========================================="
    echo "[Phase 3] Not yet implemented."
    echo "[Phase 3] Will cover: bottleneck profiling, mitigation design."
    echo ""
    echo "[Phase 3] Skipped."
}

# ===================================================================
# Dispatch
# ===================================================================
case "$PHASE" in
    test)
        run_phase_test
        ;;
    1)
        run_phase_1
        ;;
    2)
        run_phase_2
        ;;
    3)
        run_phase_3
        ;;
    all)
        run_phase_test
        echo ""
        read -p "Press Enter to continue to Phase 1 (ensure model weights are downloaded)..."
        run_phase_1
        echo ""
        read -p "Press Enter to continue to Phase 2..."
        run_phase_2
        echo ""
        read -p "Press Enter to continue to Phase 3..."
        run_phase_3
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid options: test, 1, 2, 3, all"
        exit 1
        ;;
esac

# -------------------------------------------------------------------
# Cleanup: stop vLLM server unless --keep-vllm
# -------------------------------------------------------------------
if [ -n "${VLLM_PID_FILE:-}" ] && [ -f "$VLLM_PID_FILE" ]; then
    VLLM_PID=$(cat "$VLLM_PID_FILE")
    if $KEEP_VLLM; then
        echo ""
        echo "[Cleanup] --keep-vllm: vLLM server left running (PID: $VLLM_PID)"
    else
        echo ""
        echo "[Cleanup] Stopping vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        rm -f "$VLLM_PID_FILE"
        echo "[Cleanup] vLLM server stopped."
    fi
fi

echo ""
echo "============================================"
echo " Benchmark pipeline complete!"
echo " Results: ./result/"
echo " Reports: ./PHASE_1_SUMMARY.md, ./PHASE_2_SUMMARY.md"
echo "============================================"
