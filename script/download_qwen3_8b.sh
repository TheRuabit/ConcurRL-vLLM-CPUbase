#!/bin/bash
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
exec ./venv_18/bin/huggingface-cli download Qwen/Qwen3-8B \
  --local-dir Qwen/Qwen3-8B \
  --resume-download
