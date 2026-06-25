"""
02_launch_vllm.py - Production vLLM Engine Deployment

Configures and launches the vLLM backend for dual A800 GPU setup.
Uses subprocess to spawn vLLM's OpenAI-compatible API server.

Hardcoded targets:
  --model Qwen/Qwen3-30B-A3B
  --tensor-parallel-size 2
  --max-model-len 4096
  --enable-chunked-prefill true
"""

import argparse
import subprocess
import sys
import time

DEFAULT_CONFIG = {
    "model": "Qwen/Qwen3-30B-A3B",
    "tensor_parallel_size": 2,
    "max_model_len": 4096,
    "enable_chunked_prefill": True,
    "host": "0.0.0.0",
    "port": 8000,
    "gpu_memory_utilization": 0.92,
    "dtype": "auto",
}


def build_command(config: dict) -> list[str]:
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    for key, value in config.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch vLLM server")
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"])
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG["port"])
    parser.add_argument("--tp", type=int, default=DEFAULT_CONFIG["tensor_parallel_size"])
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_CONFIG["max_model_len"])
    parser.add_argument("--gpu-mem-util", type=float, default=DEFAULT_CONFIG["gpu_memory_utilization"])
    args = parser.parse_args()

    config = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "enable_chunked_prefill": True,
        "host": DEFAULT_CONFIG["host"],
        "port": args.port,
        "gpu_memory_utilization": args.gpu_mem_util,
        "dtype": DEFAULT_CONFIG["dtype"],
    }

    cmd = build_command(config)

    print("=" * 60)
    print("  ConcurRL-vLLM Engine Launcher")
    print("=" * 60)
    print(f"\n  Model:       {config['model']}")
    print(f"  TP Size:     {config['tensor_parallel_size']}")
    print(f"  Max Seq Len: {config['max_model_len']}")
    print(f"  GPU Mem:     {config['gpu_memory_utilization']*100:.0f}%")
    print(f"  Port:        {config['port']}")
    print(f"\n  Command: {' '.join(cmd)}\n")
    print("=" * 60)

    print("\n[*] Launching vLLM server...")
    t0 = time.time()

    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[*] Shutting down vLLM server...")
        proc.terminate()
        proc.wait(timeout=10)
        elapsed = time.time() - t0
        print(f"[*] Server ran for {elapsed:.1f}s. Goodbye.")
        sys.exit(0)

    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\n[ERROR] vLLM exited with code {proc.returncode} after {elapsed:.1f}s")
        sys.exit(proc.returncode)
    else:
        print(f"\n[*] vLLM exited cleanly after {elapsed:.1f}s")


if __name__ == "__main__":
    main()
