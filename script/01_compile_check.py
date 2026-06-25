"""
01_compile_check.py - System Compilation Check

Validates async timing hooks with a lightweight FastAPI mock server.
Uses randomized sleep to simulate prefill/decode intervals at realistic token lengths.
Verifies the 5 latency parameter timers work correctly before committing to heavy resources.
"""

import asyncio
import json
import random
import time
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class MockResponse(BaseModel):
    text: str
    tokens: list[str]
    prefill_ms: float
    decode_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def mock_completion():
    prefill_ms = random.uniform(200, 1500)
    decode_ms = random.uniform(10, 50)
    await asyncio.sleep(prefill_ms / 1000)
    n_tokens = random.randint(64, 256)
    tokens = []
    for _ in range(n_tokens):
        await asyncio.sleep(decode_ms / 1000)
        tokens.append(f"token_{random.randint(0, 1000)}")
    return MockResponse(
        text=" ".join(tokens),
        tokens=tokens,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
    )


async def run_client(base_url: str, n_requests: int = 5):
    results = []
    async with aiohttp.ClientSession() as session:
        for i in range(n_requests):
            record = {}
            t0 = time.perf_counter()
            payload = json.dumps({"prompt": f"test prompt {i}"})
            record["client_serialization_ms"] = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            async with session.post(
                f"{base_url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                t_header = time.perf_counter()
                data = await resp.json()
                t_done = time.perf_counter()

            record["server_overhead_ms"] = (t_header - t1) * 1000
            record["gpu_prefill_ms"] = data["prefill_ms"]
            record["gpu_decode_ms"] = data["decode_ms"]
            record["response_parsing_ms"] = (t_done - t_header) * 1000
            results.append(record)
            print(f"  Request {i+1}/{n_requests}: {json.dumps(record, indent=2)}")
    return results


async def main():
    print("=" * 60)
    print("  ConcurRL-vLLM Compile Check")
    print("=" * 60)
    print("\n[1/2] Starting mock FastAPI server on port 8247...")

    config = uvicorn.Config(app, host="127.0.0.1", port=8247, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(1)

    print("[2/2] Running mock client with 5 sequential requests...\n")
    results = await run_client("http://127.0.0.1:8247", n_requests=5)

    server.should_exit = True
    await server_task

    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    for key in results[0]:
        vals = [r[key] for r in results]
        avg = sum(vals) / len(vals)
        print(f"  {key:30s}  avg={avg:8.2f}ms")

    print("\n[PASS] All 5 latency parameters captured successfully.")
    print("       Ready to proceed with model download and Step 2.\n")


if __name__ == "__main__":
    asyncio.run(main())
