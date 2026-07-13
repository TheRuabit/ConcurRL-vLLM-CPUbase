#!/bin/bash
# Start standalone vLLM server on GPU 0,1 for Phase 2 external mode
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export VLLM_SCHED_TRACE="${VLLM_SCHED_TRACE:-1}"
export VLLM_SCHED_TRACE_PATH="${VLLM_SCHED_TRACE_PATH:-$PWD/result/vllm_scheduler_trace.jsonl}"
export VLLM_MODEL="${VLLM_MODEL:-/mnt/data1/kwchen/yschen/ConcurRL-vLLM-CPUbase/Qwen/Qwen3-8B}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1024}"
export VLLM_PORT="${VLLM_PORT:-8000}"

mkdir -p "$(dirname "$VLLM_SCHED_TRACE_PATH")"

exec ./.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL" \
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --port "$VLLM_PORT" \
  --trust-remote-code \
  --enforce-eager \
  --enable-prefix-caching
