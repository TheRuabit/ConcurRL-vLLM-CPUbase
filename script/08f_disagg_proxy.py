#!/usr/bin/env python3
"""
Phase 2f: Disaggregated Prefill Proxy Server (aiohttp-based)
=============================================================
Routes requests through a disaggregated prefill/decode pipeline.
All requests go to the prefill instance, which handles KV transfer
to the decode instance internally via NIXL.

Usage:
    python script/08f_disagg_proxy.py
    python script/08f_disagg_proxy.py --prefill-url http://localhost:8100 --decode-url http://localhost:8200
"""

import argparse
import asyncio
from pathlib import Path

parser = argparse.ArgumentParser(description="Disaggregated prefill proxy server")
parser.add_argument("--prefill-url", default="http://localhost:8100",
                    help="Prefill vLLM instance URL")
parser.add_argument("--decode-url", default="http://localhost:8200",
                    help="Decode vLLM instance URL")
parser.add_argument("--port", type=int, default=8000,
                    help="Proxy listening port")
parser.add_argument("--host", default="0.0.0.0",
                    help="Proxy listening host")
args = parser.parse_args()

try:
    from aiohttp import web, ClientSession, ClientTimeout
except ImportError:
    print("[proxy] ERROR: aiohttp required. Install: pip install aiohttp")
    exit(1)

PREFILL_URL = args.prefill_url.rstrip("/")
DECODE_URL = args.decode_url.rstrip("/")


async def health_handler(request):
    """Check health of both prefill and decode instances."""
    async with ClientSession(timeout=ClientTimeout(total=5)) as session:
        prefill_ok = False
        decode_ok = False
        try:
            async with session.get(f"{PREFILL_URL}/health") as resp:
                prefill_ok = resp.status == 200
        except Exception:
            pass
        try:
            async with session.get(f"{DECODE_URL}/health") as resp:
                decode_ok = resp.status == 200
        except Exception:
            pass

    if prefill_ok and decode_ok:
        return web.json_response({"status": "ok", "prefill": "ok", "decode": "ok"})
    else:
        return web.json_response(
            {"status": "degraded", "prefill": "ok" if prefill_ok else "down",
             "decode": "ok" if decode_ok else "down"},
            status=503
        )


async def metrics_handler(request):
    """Proxy metrics from prefill instance."""
    async with ClientSession(timeout=ClientTimeout(total=5)) as session:
        try:
            async with session.get(f"{PREFILL_URL}/metrics") as resp:
                text = await resp.text()
                return web.Response(text=text, content_type="text/plain")
        except Exception:
            return web.json_response({"error": "prefill unreachable"}, status=502)


async def chat_completions_handler(request):
    """Route chat completions through disaggregated prefill pipeline."""
    body = await request.read()
    headers = {}
    for k, v in request.headers.items():
        if k.lower() not in ("host", "content-length"):
            headers[k] = v

    target_url = f"{PREFILL_URL}/v1/chat/completions"

    session = ClientSession(timeout=ClientTimeout(total=600))
    try:
        async with session.post(target_url, data=body, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                await session.close()
                return web.json_response(
                    {"error": error_body[:500]},
                    status=resp.status
                )

            # Stream response back to client
            response = web.StreamResponse(
                status=resp.status,
                headers={
                    "Content-Type": "text/event-stream",
                    "X-Disagg-Prefill": "true",
                },
            )
            await response.prepare(request)

            async for chunk in resp.content.iter_any():
                await response.write(chunk)

            await response.write_eof()
            await session.close()
            return response

    except Exception as e:
        await session.close()
        return web.json_response({"error": str(e)[:300]}, status=500)


async def completions_handler(request):
    """Route completions through disaggregated prefill pipeline."""
    body = await request.read()
    headers = {}
    for k, v in request.headers.items():
        if k.lower() not in ("host", "content-length"):
            headers[k] = v

    target_url = f"{PREFILL_URL}/v1/completions"

    session = ClientSession(timeout=ClientTimeout(total=600))
    try:
        async with session.post(target_url, data=body, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                await session.close()
                return web.json_response(
                    {"error": error_body[:500]},
                    status=resp.status
                )

            response = web.StreamResponse(
                status=resp.status,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)

            async for chunk in resp.content.iter_any():
                await response.write(chunk)

            await response.write_eof()
            await session.close()
            return response

    except Exception as e:
        await session.close()
        return web.json_response({"error": str(e)[:300]}, status=500)


app = web.Application()
app.router.add_get("/health", health_handler)
app.router.add_get("/metrics", metrics_handler)
app.router.add_post("/v1/chat/completions", chat_completions_handler)
app.router.add_post("/v1/completions", completions_handler)

if __name__ == "__main__":
    print(f"[proxy] Disaggregated Prefill Proxy (aiohttp)")
    print(f"  Prefill: {PREFILL_URL}")
    print(f"  Decode:  {DECODE_URL}")
    print(f"  Listen:  {args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=None)
