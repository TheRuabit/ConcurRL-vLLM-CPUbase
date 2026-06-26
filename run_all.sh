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
#   Phase 2 — Full RL Loop Integration    [TODO]
#   Phase 3 — Analysis & Solution Design  [TODO]
#
# Usage:
#   bash run_all.sh --phase test                          # Compile check only
#   bash run_all.sh --phase 1                             # Full Phase 1 pipeline
#   bash run_all.sh --phase all                            # Everything
#   bash run_all.sh --phase 1 --model ./models/Qwen3-30B-A3B
#   bash run_all.sh --phase 1 --max_output_token 128
#   bash run_all.sh --phase 1 --url http://localhost:8000
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
        -h|--help)
            echo "Usage: bash run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --phase <test|1|all>    Which phase to run (default: all)"
            echo "                          test  = 01_compile_check.py (mock server)"
            echo "                          1     = 02+03+04 (vLLM concurrency sweep)"
            echo "                          all   = all phases"
            echo "  --model <path>          Model name or path (default: Qwen/Qwen3-30B-A3B)"
            echo "  --max_output_token <N>  Max output tokens per request (default: 64)"
            echo "  --url <url>             vLLM server URL (default: http://localhost:8000)"
            echo "  --python <cmd>          Python interpreter (default: python)"
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
if [ -f ".venv/bin/activate" ]; then
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
run_phase_1() {
    echo "==========================================="
    echo " PHASE 1: Concurrency Scaling Validation"
    echo "==========================================="

    # Check if server is already running
    echo "[Phase1] Checking server at $SERVER_URL ..."
    SERVER_RUNNING=false
    if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo "[Phase1] Server already running."
        SERVER_RUNNING=true
    fi

    # Launch vLLM server if not running
    if ! $SERVER_RUNNING; then
        echo ""
        echo "--- 02 Launch vLLM Server ---"
        echo "[Phase1] Starting vLLM server (this may take a few minutes)..."
        $PYTHON script/02_launch_vllm.py \
            --model "$MODEL" \
            --port "${SERVER_URL##*:}" \
            &
        VLLM_PID=$!
        echo "[Phase1] vLLM server PID: $VLLM_PID"

        # Wait for health
        echo "[Phase1] Waiting for server to become healthy..."
        for i in $(seq 1 60); do
            if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
                echo "[Phase1] Server ready after ${i}x5s."
                break
            fi
            if [ "$i" -eq 60 ]; then
                echo "[Phase1] ERROR: Server did not become healthy in 300s."
                kill $VLLM_PID 2>/dev/null || true
                exit 1
            fi
            sleep 5
        done
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

    # Clean up vLLM server if we launched it
    if [ -n "${VLLM_PID:-}" ]; then
        echo ""
        echo "[Phase1] Stopping vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
    fi

    echo ""
    echo "[Phase 1] Complete. Results in ./result/ and PHASE_1_SUMMARY.md"
}

# ===================================================================
# Phase 2: Full RL Loop Integration [TODO]
# ===================================================================
run_phase_2() {
    echo "==========================================="
    echo " PHASE 2: Full RL Loop Integration"
    echo "==========================================="
    echo "[Phase 2] Not yet implemented."
    echo "[Phase 2] Will cover: alternation mechanics, phase transition bottlenecks."
    echo ""
    echo "[Phase 2] Skipped."
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

echo ""
echo "============================================"
echo " Benchmark pipeline complete!"
echo " Results: ./result/"
echo " Report:  ./PHASE_1_SUMMARY.md"
echo "============================================"
