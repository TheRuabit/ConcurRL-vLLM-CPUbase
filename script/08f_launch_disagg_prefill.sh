#!/bin/bash
# ==============================================================================
# Phase 2f: Disaggregated Prefill Launch Script
# ==============================================================================
# Launches 2 vLLM instances with NixlConnector for disaggregated prefill/decode,
# plus a proxy server that routes requests through the pipeline.
#
# Architecture:
#   Client → Proxy (port 8000) → Prefill vLLM (port 8100, GPU 0)
#                              → Decode vLLM  (port 8200, GPU 1)
#   KV cache transferred via NIXL between prefill and decode instances
#
# Usage:
#   bash script/08f_launch_disagg_prefill.sh
#   bash script/08f_launch_disagg_prefill.sh --model Qwen/Qwen3-30B-A3B
#   bash script/08f_launch_disagg_prefill.sh --detach --pid-file result/disagg_pids.txt
#
# Prerequisites:
#   - 2 GPUs (uses GPU 0 for prefill, GPU 1 for decode)
#   - nixl package installed (pip install nixl)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
MODEL="Qwen/Qwen3-30B-A3B"
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.85
HEALTH_TIMEOUT=300
DETACH=false
PID_FILE=""
LOG_DIR="$PROJECT_DIR/result/disagg_logs"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --prefill-port) PREFILL_PORT="$2"; shift 2 ;;
        --decode-port) DECODE_PORT="$2"; shift 2 ;;
        --proxy-port) PROXY_PORT="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
        --health-timeout) HEALTH_TIMEOUT="$2"; shift 2 ;;
        --detach) DETACH=true; shift ;;
        --pid-file) PID_FILE="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash script/08f_launch_disagg_prefill.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model <path>          Model name (default: Qwen/Qwen3-30B-A3B)"
            echo "  --prefill-port <N>      Prefill vLLM port (default: 8100)"
            echo "  --decode-port <N>       Decode vLLM port (default: 8200)"
            echo "  --proxy-port <N>        Proxy server port (default: 8000)"
            echo "  --max-model-len <N>     Max model context length (default: 32768)"
            echo "  --gpu-mem-util <F>      GPU memory utilization (default: 0.85)"
            echo "  --health-timeout <N>    Health check timeout in seconds (default: 300)"
            echo "  --detach                Launch in background, write PID file"
            echo "  --pid-file <path>       Write PIDs to file (used with --detach)"
            echo "  --log-dir <path>        Log directory (default: result/disagg_logs)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Activate venv
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

PYTHON=$(which python)
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Phase 2f: Disaggregated Prefill Launch"
echo "============================================================"
echo "  Model:          $MODEL"
echo "  Prefill GPU:    0 (port $PREFILL_PORT)"
echo "  Decode GPU:     1 (port $DECODE_PORT)"
echo "  Proxy:          localhost:$PROXY_PORT"
echo "  Max model len:  $MAX_MODEL_LEN"
echo "  GPU mem util:   $GPU_MEM_UTIL"
echo "============================================================"

# -------------------------------------------------------------------
# Kill existing servers on these ports
# -------------------------------------------------------------------
for port in $PREFILL_PORT $DECODE_PORT $PROXY_PORT; do
    pid=$(lsof -ti :$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "[cleanup] Killing existing process on port $port (PID: $pid)"
        kill $pid 2>/dev/null || true
        sleep 1
    fi
done

# -------------------------------------------------------------------
# Launch Prefill vLLM instance (GPU 0)
# -------------------------------------------------------------------
echo ""
echo "[1/3] Launching Prefill vLLM instance on GPU 0 (port $PREFILL_PORT)..."
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port $PREFILL_PORT \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --enable-chunked-prefill \
    --max-num-seqs 1024 \
    --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":14579}" \
    > "$LOG_DIR/prefill.log" 2>&1 &
PREFILL_PID=$!
echo "  Prefill PID: $PREFILL_PID"

# Wait for prefill to be healthy BEFORE starting decode
echo -n "  Waiting for Prefill (port $PREFILL_PORT): "
for i in $(seq 1 $HEALTH_TIMEOUT); do
    if curl -sf "http://localhost:$PREFILL_PORT/health" > /dev/null 2>&1; then
        echo "ready (${i}s)"
        break
    fi
    if [ $i -eq $HEALTH_TIMEOUT ]; then
        echo "TIMEOUT after ${HEALTH_TIMEOUT}s"
        echo "[ERROR] Prefill failed to start. Check $LOG_DIR/prefill.log"
        kill $PREFILL_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# -------------------------------------------------------------------
# Launch Decode vLLM instance (GPU 1)
# -------------------------------------------------------------------
echo ""
echo "[2/3] Launching Decode vLLM instance on GPU 1 (port $DECODE_PORT)..."
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port $DECODE_PORT \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --enable-chunked-prefill \
    --max-num-seqs 1024 \
    --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_port\":14580}" \
    > "$LOG_DIR/decode.log" 2>&1 &
DECODE_PID=$!
echo "  Decode PID: $DECODE_PID"

# Wait for decode to be healthy
echo -n "  Waiting for Decode (port $DECODE_PORT): "
for i in $(seq 1 $HEALTH_TIMEOUT); do
    if curl -sf "http://localhost:$DECODE_PORT/health" > /dev/null 2>&1; then
        echo "ready (${i}s)"
        break
    fi
    if [ $i -eq $HEALTH_TIMEOUT ]; then
        echo "TIMEOUT after ${HEALTH_TIMEOUT}s"
        echo "[ERROR] Decode failed to start. Check $LOG_DIR/decode.log"
        kill $PREFILL_PID $DECODE_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# -------------------------------------------------------------------
# Launch Proxy server
# -------------------------------------------------------------------
echo ""
echo "[3/3] Launching Proxy server on port $PROXY_PORT..."
$PYTHON "$SCRIPT_DIR/08f_disagg_proxy.py" \
    --prefill-url "http://localhost:$PREFILL_PORT" \
    --decode-url "http://localhost:$DECODE_PORT" \
    --port $PROXY_PORT \
    > "$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
echo "  Proxy PID: $PROXY_PID"

# Wait for proxy
echo -n "  Proxy: "
for i in $(seq 1 30); do
    if curl -sf "http://localhost:$PROXY_PORT/health" > /dev/null 2>&1; then
        echo "ready (${i}s)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "TIMEOUT"
        kill $PREFILL_PID $DECODE_PID $PROXY_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# -------------------------------------------------------------------
# Write PID file
# -------------------------------------------------------------------
if [ -n "$PID_FILE" ]; then
    echo "$PREFILL_PID $DECODE_PID $PROXY_PID" > "$PID_FILE"
    echo ""
    echo "PIDs written to $PID_FILE"
fi

echo ""
echo "============================================================"
echo " Disaggregated Prefill setup ready!"
echo "   Proxy:     http://localhost:$PROXY_PORT"
echo "   Prefill:   http://localhost:$PREFILL_PORT"
echo "   Decode:    http://localhost:$DECODE_PORT"
echo "   Logs:      $LOG_DIR/"
echo ""
echo " Run benchmark:"
echo "   python script/08_tool_call_driver.py --url http://localhost:$PROXY_PORT ..."
echo "============================================================"

# If not detach, wait for children
if ! $DETACH; then
    echo ""
    echo "Press Ctrl+C to stop all servers..."
    trap "kill $PREFILL_PID $DECODE_PID $PROXY_PID 2>/dev/null; exit 0" INT TERM
    wait
fi
