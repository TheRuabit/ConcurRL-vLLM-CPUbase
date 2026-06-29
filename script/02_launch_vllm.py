#!/usr/bin/env python3
"""
Production vLLM Engine Launcher
================================
Spawns a vLLM server via subprocess with optimal settings for dual-GPU
(A800-80GB x2) tensor parallelism. Waits for health check before exiting.

Hardcoded targets (Phase1Plan.md Step 2):
  --model Qwen/Qwen3-30B-A3B
  --tensor-parallel-size 2
  --max-model-len 4096
  --enable-chunked-prefill true

Usage:
    python script/02_launch_vllm.py
    python script/02_launch_vllm.py --model ./models/Qwen3-30B-A3B --port 8000
    python script/02_launch_vllm.py --wait-only   # Just wait for existing server

Output:
    Launches vLLM server in foreground. Ctrl+C to stop.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Launch vLLM server for Phase 1 benchmarking")
parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                    help="Model name or path")
parser.add_argument("--port", type=int, default=8000,
                    help="Server port")
parser.add_argument("--host", default="0.0.0.0",
                    help="Server host")
parser.add_argument("--tensor-parallel-size", type=int, default=2,
                    help="Number of GPUs for tensor parallelism")
parser.add_argument("--max-model-len", type=int, default=51200,
                    help="Maximum model context length")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                    help="GPU memory utilization fraction")
parser.add_argument("--enable-chunked-prefill", action="store_true", default=True,
                    help="Enable chunked prefill (recommended for high concurrency)")
parser.add_argument("--no-chunked-prefill", action="store_true",
                    help="Disable chunked prefill")
parser.add_argument("--dtype", default="auto",
                    help="Model dtype (auto, float16, bfloat16)")
parser.add_argument("--max-num-seqs", type=int, default=1024,
                    help="Maximum number of sequences per iteration")
parser.add_argument("--wait-only", action="store_true",
                    help="Only wait for an existing server to become healthy")
parser.add_argument("--detach", action="store_true",
                    help="Launch server, wait for health, write PID file, then exit "
                         "(for use by run_all.sh)")
parser.add_argument("--pid-file", default=None,
                    help="Write vLLM subprocess PID to this file (used with --detach)")
parser.add_argument("--health-timeout", type=int, default=200,
                    help="Seconds to wait for server health check")
parser.add_argument("--log-file", default=None,
                    help="Redirect vLLM output to this file")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEALTH_URL = f"http://127.0.0.1:{args.port}/health"
API_URL = f"http://127.0.0.1:{args.port}/v1/models"


def wait_for_health(url: str, timeout: int) -> bool:
    """Poll health endpoint until server is ready."""
    import urllib.request
    import urllib.error

    print(f"[02_launch] Waiting for server at {url}...")
    start = time.perf_counter()
    attempt = 0
    while time.perf_counter() - start < timeout:
        attempt += 1
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    elapsed = time.perf_counter() - start
                    print(f"[02_launch] Server ready after {elapsed:.1f}s (attempt {attempt})")
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        if attempt % 10 == 0:
            elapsed = time.perf_counter() - start
            print(f"[02_launch] Still waiting... ({elapsed:.0f}s elapsed, attempt {attempt})")
        time.sleep(2.0)

    print(f"[02_launch] ERROR: Server not healthy after {timeout}s")
    return False


def main():
    print(f"[02_launch] vLLM Server Launcher")
    print(f"[02_launch] Model:  {args.model}")
    print(f"[02_launch] Port:   {args.port}")
    print(f"[02_launch] TP:     {args.tensor_parallel_size}")
    print(f"[02_launch] MaxLen: {args.max_model_len}")
    print()

    if args.wait_only:
        ok = wait_for_health(HEALTH_URL, args.health_timeout)
        sys.exit(0 if ok else 1)

    # Build vLLM command
    chunked = not args.no_chunked_prefill
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype,
        "--max-num-seqs", str(args.max_num_seqs),
    ]
    if chunked:
        cmd.append("--enable-chunked-prefill")

    print(f"[02_launch] Command: {' '.join(cmd)}")
    print(f"[02_launch] Starting vLLM server...\n")

    # Prepare log file
    log_handle = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "w")
        print(f"[02_launch] Logging to: {log_path}")

    # Launch subprocess
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=log_handle if log_handle else sys.stdout,
            stderr=subprocess.STDOUT if log_handle else sys.stderr,
            text=True,
        )

        # Wait for health in a background thread while process runs
        ok = wait_for_health(HEALTH_URL, args.health_timeout)
        if not ok:
            print("[02_launch] Server failed to start, terminating...")
            process.terminate()
            process.wait(timeout=10)
            sys.exit(1)

        # Detach mode: write PID file and exit (caller manages lifecycle)
        if args.detach:
            pid_path = Path(args.pid_file) if args.pid_file else \
                       Path(__file__).resolve().parents[1] / "result" / "vllm_server.pid"
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text(str(process.pid), encoding="utf-8")
            print(f"[02_launch] vLLM server running at http://{args.host}:{args.port}")
            print(f"[02_launch] PID: {process.pid} (written to {pid_path})")
            # Do NOT wait — let caller manage the server
            if log_handle:
                log_handle.close()
            return

        print(f"\n[02_launch] vLLM server running at http://{args.host}:{args.port}")
        print(f"[02_launch] Press Ctrl+C to stop\n")

        # Block until process exits or Ctrl+C
        process.wait()

    except KeyboardInterrupt:
        print("\n[02_launch] Shutting down server...")
        if process:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print("[02_launch] Force killing server...")
                process.kill()
                process.wait()
        print("[02_launch] Server stopped.")
    finally:
        if log_handle:
            log_handle.close()


if __name__ == "__main__":
    main()
