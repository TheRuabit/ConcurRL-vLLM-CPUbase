#!/usr/bin/env python3
"""veRL group rollout-only execution and request-correlated vLLM tracing.

This module deliberately uses Python-level patches only.  The patches are
installed inside the relevant Ray actors before vLLM forks its engine core, so
the scheduler/KV-cache patch is inherited without rebuilding vLLM or CUDA code.
"""

from __future__ import annotations

import contextvars
import enum
import json
import math
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


# The pinned veRL commit advertises Python >=3.10 but imports enum.StrEnum,
# which only became part of the stdlib in 3.11.
if not hasattr(enum, "StrEnum"):
    class _StrEnum(str, enum.Enum):
        pass

    enum.StrEnum = _StrEnum


TRACE_ENABLED_ENV = "CONCURL_ROLLOUT_TRACE"
TRACE_PATH_ENV = "CONCURL_ROLLOUT_TRACE_PATH"

_request_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "concurl_request_context", default=None
)
_agent_patched = False
_vllm_patched = False
_driver_patched = False


def _enabled() -> bool:
    return os.getenv(TRACE_ENABLED_ENV, "1").lower() not in {"0", "false", "no", "off"}


def _emit(event: str, *, timestamp: float | None = None, **fields: Any) -> None:
    """Append one small JSON event atomically enough for local multi-process use."""
    if not _enabled():
        return
    trace_path = os.getenv(TRACE_PATH_ENV)
    if not trace_path:
        return
    record = {
        "event": event,
        "timestamp": time.monotonic() if timestamp is None else float(timestamp),
        "pid": os.getpid(),
        **fields,
    }
    payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    fd = os.open(trace_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def install_agent_worker_patches() -> None:
    """Install patches in each veRL AgentLoopWorker process."""
    global _agent_patched
    if _agent_patched:
        return

    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    import verl.workers.rollout.llm_server as llm_server_module

    original_run = SingleTurnAgentLoop.run
    original_uuid4 = llm_server_module.uuid4
    original_generate = llm_server_module.LLMServerClient.generate

    async def traced_run(self, sampling_params, priority=0, **kwargs):
        context = {
            "group_id": str(kwargs.get("uid", "unknown")),
            "sample_index": int(priority),
            "rollout_step": int(kwargs.get("trace_step", -1)),
            "phase": str(kwargs.get("trace_phase", "measured")),
            "prep_start": time.monotonic(),
            "turn": 0,
        }
        token = _request_context.set(context)
        try:
            return await original_run(self, sampling_params, priority=priority, **kwargs)
        finally:
            _request_context.reset(token)

    class _RequestId:
        def __init__(self, value: str):
            self.hex = value

    def traced_uuid4():
        context = _request_context.get()
        if context is None:
            return original_uuid4()
        context["turn"] += 1
        request_id = f"cr-{uuid.uuid4().hex}"
        context["request_id"] = request_id
        _emit(
            "request_submit",
            request_id=request_id,
            group_id=context["group_id"],
            sample_index=context["sample_index"],
            rollout_step=context["rollout_step"],
            phase=context["phase"],
            prep_start=context["prep_start"],
            turn=context["turn"],
        )
        return _RequestId(request_id)

    async def traced_generate(self, *args, **kwargs):
        context = _request_context.get()
        try:
            output = await original_generate(self, *args, **kwargs)
        except Exception as exc:
            _emit(
                "request_terminal_failure",
                request_id=(context or {}).get("request_id"),
                group_id=(context or {}).get("group_id"),
                sample_index=(context or {}).get("sample_index"),
                rollout_step=(context or {}).get("rollout_step"),
                phase=(context or {}).get("phase"),
                error=repr(exc)[:500],
            )
            raise
        _emit(
            "request_complete",
            request_id=(context or {}).get("request_id"),
            group_id=(context or {}).get("group_id"),
            sample_index=(context or {}).get("sample_index"),
            rollout_step=(context or {}).get("rollout_step"),
            phase=(context or {}).get("phase"),
            output_tokens=len(output.token_ids or []),
            stop_reason=str(getattr(output, "stop_reason", "")),
        )
        return output

    SingleTurnAgentLoop.run = traced_run
    llm_server_module.uuid4 = traced_uuid4
    llm_server_module.LLMServerClient.generate = traced_generate
    _agent_patched = True


def install_vllm_patches() -> None:
    """Install vLLM 0.11 V1 scheduler/frontend patches before engine startup."""
    global _vllm_patched
    if _vllm_patched:
        return

    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.metrics.stats import IterationStats
    except ImportError as exc:
        raise RuntimeError("rollout tracing requires the vLLM 0.11 V1 engine") from exc

    original_allocate_slots = KVCacheManager.allocate_slots
    original_update_from_events = IterationStats.update_from_events
    original_update_from_output = IterationStats.update_from_output
    seen_first_output: set[str] = set()

    def traced_allocate_slots(self, request, *args, **kwargs):
        blocks = original_allocate_slots(self, request, *args, **kwargs)
        if blocks is None:
            _emit(
                "kv_allocation_failure",
                request_id=request.request_id,
                request_status=str(request.status),
            )
        return blocks

    def traced_update_from_events(self, req_id, events, *args, **kwargs):
        result = original_update_from_events(self, req_id, events, *args, **kwargs)
        from vllm.v1.engine import EngineCoreEventType

        for core_event in events:
            if core_event.type == EngineCoreEventType.QUEUED:
                _emit("engine_queued", timestamp=core_event.timestamp, request_id=req_id)
            elif core_event.type == EngineCoreEventType.SCHEDULED:
                _emit("engine_scheduled", timestamp=core_event.timestamp, request_id=req_id)
            elif core_event.type == EngineCoreEventType.PREEMPTED:
                _emit(
                    "request_preempted",
                    timestamp=core_event.timestamp,
                    request_id=req_id,
                    reason=None,
                )
        return result

    def traced_update_from_output(
        self,
        output,
        engine_core_timestamp,
        is_prefilling,
        *args,
        **kwargs,
    ):
        result = original_update_from_output(
            self,
            output,
            engine_core_timestamp,
            is_prefilling,
            *args,
            **kwargs,
        )
        if is_prefilling and output.request_id not in seen_first_output:
            seen_first_output.add(output.request_id)
            _emit(
                "first_output",
                timestamp=engine_core_timestamp,
                request_id=output.request_id,
                new_tokens=len(output.new_token_ids),
            )
        return result

    KVCacheManager.allocate_slots = traced_allocate_slots
    IterationStats.update_from_events = traced_update_from_events
    IterationStats.update_from_output = traced_update_from_output
    _vllm_patched = True


def install_driver_patches(*, warmup_steps: int, measured_steps: int) -> None:
    """Install the server-actor hook and replace PPO fit with rollout-only fit."""
    global _driver_patched
    if _driver_patched:
        return

    import numpy as np
    import ray
    import verl.workers.rollout.vllm_rollout.vllm_async_server as server_module
    from verl.protocol import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    base_server = server_module.vLLMHttpServer

    class InstrumentedVLLMHttpServer(base_server):
        def __init__(self, *args, **kwargs):
            install_vllm_patches()
            super().__init__(*args, **kwargs)

    server_module.vLLMHttpServer = InstrumentedVLLMHttpServer

    def rollout_only_fit(self):
        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        total_steps = warmup_steps + measured_steps
        data_iterator = iter(self.train_dataloader)
        completed = 0
        for rollout_step in range(total_steps):
            try:
                batch_dict = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader)
                batch_dict = next(data_iterator)

            if rollout_step > 0:
                self.checkpoint_manager.wake_up_replicas()

            batch = DataProto.from_single_dict(batch_dict)
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
            gen_batch = self._get_gen_batch(batch)
            gen_batch.meta_info["global_steps"] = rollout_step
            gen_batch.non_tensor_batch["trace_step"] = np.full(
                len(gen_batch), rollout_step, dtype=np.int64
            )
            phase = "warmup" if rollout_step < warmup_steps else "measured"
            gen_batch.non_tensor_batch["trace_phase"] = np.array(
                [phase] * len(gen_batch), dtype=object
            )
            repeated = gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n,
                interleave=True,
            )

            _emit(
                "rollout_step_start",
                rollout_step=rollout_step,
                phase=phase,
                groups=len(gen_batch),
                requests=len(repeated),
            )
            output = self.async_rollout_manager.generate_sequences(repeated)
            self.checkpoint_manager.sleep_replicas()
            _emit(
                "rollout_step_complete",
                rollout_step=rollout_step,
                phase=phase,
                requests=len(output),
            )
            completed += 1
            del output, repeated, gen_batch, batch

        _emit(
            "rollout_only_complete",
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            total_steps=completed,
        )

    RayPPOTrainer.fit = rollout_only_fit
    _driver_patched = True


class InstrumentedAgentLoopWorker:
    """Factory placeholder replaced with the real veRL subclass at import time."""


class InstrumentedAgentLoopManager:
    """Factory placeholder replaced with the real veRL subclass at import time."""


def _define_agent_classes() -> None:
    import ray
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker

    class _InstrumentedAgentLoopWorker(AgentLoopWorker):
        def __init__(self, *args, **kwargs):
            install_agent_worker_patches()
            super().__init__(*args, **kwargs)

    class _InstrumentedAgentLoopManager(AgentLoopManager):
        def __init__(self, *args, **kwargs):
            self.agent_loop_workers_class = ray.remote(_InstrumentedAgentLoopWorker)
            super().__init__(*args, **kwargs)

    global InstrumentedAgentLoopWorker, InstrumentedAgentLoopManager
    InstrumentedAgentLoopWorker = _InstrumentedAgentLoopWorker
    InstrumentedAgentLoopManager = _InstrumentedAgentLoopManager


_define_agent_classes()


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


def _stage_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50_ms": _nearest_rank(values, 50),
        "p95_ms": _nearest_rank(values, 95),
        "p99_ms": _nearest_rank(values, 99),
        "mean_ms": (sum(values) / len(values)) if values else None,
    }


def aggregate_trace(trace_path: str | Path, *, expected_group_size: int) -> dict[str, Any]:
    """Build measured-step request/group metrics from the shared JSONL trace."""
    path = Path(trace_path)
    events: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    requests: dict[str, dict[str, Any]] = defaultdict(dict)
    kv_failures = 0
    preemptions = 0
    terminal_failures = 0

    for event in events:
        request_id = event.get("request_id")
        name = event.get("event")
        if request_id:
            record = requests[request_id]
            record.setdefault("request_id", request_id)
            if name == "request_submit":
                record.update(
                    group_id=event.get("group_id"),
                    sample_index=event.get("sample_index"),
                    rollout_step=event.get("rollout_step"),
                    phase=event.get("phase"),
                    prep_start=event.get("prep_start"),
                    submit=event.get("timestamp"),
                )
            elif name == "engine_queued":
                record.setdefault("queued", event.get("timestamp"))
            elif name == "engine_scheduled":
                record.setdefault("scheduled", event.get("timestamp"))
            elif name == "first_output":
                record.setdefault("first_output", event.get("timestamp"))
            elif name == "request_complete":
                record["complete"] = event.get("timestamp")
                record["output_tokens"] = int(event.get("output_tokens", 0))
            elif name == "kv_allocation_failure":
                kv_failures += 1
                record["kv_allocation_failures"] = record.get("kv_allocation_failures", 0) + 1
            elif name == "request_preempted":
                preemptions += 1
                record["preemptions"] = record.get("preemptions", 0) + 1
            elif name == "request_terminal_failure":
                terminal_failures += 1
                record["terminal_failure"] = event.get("error", "unknown")

    measured = [record for record in requests.values() if record.get("phase") == "measured"]
    stage_values: dict[str, list[float]] = defaultdict(list)
    valid_records: list[dict[str, Any]] = []
    for record in measured:
        prep_start = record.get("prep_start")
        submit = record.get("submit")
        queued = record.get("queued")
        scheduled = record.get("scheduled")
        first_output = record.get("first_output")
        if prep_start is not None and submit is not None:
            stage_values["prompt_preparation"].append((submit - prep_start) * 1000)
        if submit is not None and queued is not None:
            stage_values["request_admission"].append((queued - submit) * 1000)
        if queued is not None and scheduled is not None:
            stage_values["scheduler_wait"].append((scheduled - queued) * 1000)
        if scheduled is not None and first_output is not None:
            stage_values["first_scheduled_to_output"].append((first_output - scheduled) * 1000)
        if prep_start is not None and first_output is not None:
            latency_ms = (first_output - prep_start) * 1000
            stage_values["rollout_first_output_wait"].append(latency_ms)
            record["rollout_first_output_wait_ms"] = latency_ms
            valid_records.append(record)

    group_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in measured:
        if record.get("group_id"):
            group_records[record["group_id"]].append(record)

    group_ready: list[float] = []
    completed_groups = 0
    for group in group_records.values():
        first_output_latencies = [
            record["rollout_first_output_wait_ms"]
            for record in group
            if "rollout_first_output_wait_ms" in record
        ]
        if len(first_output_latencies) == expected_group_size:
            group_ready.append(max(first_output_latencies))
        if len(group) == expected_group_size and all("complete" in record for record in group):
            completed_groups += 1

    completed = [record for record in measured if "complete" in record]
    window_start = min((record["prep_start"] for record in measured if "prep_start" in record), default=None)
    window_end = max((record["complete"] for record in completed), default=None)
    duration_s = (window_end - window_start) if window_start is not None and window_end is not None else 0.0
    output_tokens = sum(record.get("output_tokens", 0) for record in completed)

    kv_affected_requests = {r["request_id"] for r in measured if r.get("kv_allocation_failures", 0)}
    kv_affected_groups = {r.get("group_id") for r in measured if r.get("kv_allocation_failures", 0)}
    preempted_requests = {r["request_id"] for r in measured if r.get("preemptions", 0)}
    preempted_groups = {r.get("group_id") for r in measured if r.get("preemptions", 0)}

    return {
        "clock": "single_host_monotonic",
        "percentile_method": "nearest_rank",
        "num_events": len(events),
        "num_measured_requests": len(measured),
        "num_valid_first_output_requests": len(valid_records),
        "num_groups": len(group_records),
        "num_complete_groups": completed_groups,
        "request_stages": {name: _stage_summary(values) for name, values in stage_values.items()},
        "request_tail_latency": _stage_summary(stage_values["rollout_first_output_wait"]),
        "group_ready_latency": _stage_summary(group_ready),
        "throughput": {
            "measurement_window_s": duration_s,
            "output_tokens": output_tokens,
            "output_tokens_per_s": output_tokens / duration_s if duration_s > 0 else None,
            "completed_groups_per_s": completed_groups / duration_s if duration_s > 0 else None,
        },
        "kv_allocation": {
            "failure_attempts": kv_failures,
            "affected_requests": len(kv_affected_requests),
            "affected_groups": len(kv_affected_groups - {None}),
        },
        "preemption": {
            "events": preemptions,
            "affected_requests": len(preempted_requests),
            "affected_groups": len(preempted_groups - {None}),
            "reason_available": False,
        },
        "terminal_request_failures": terminal_failures,
        "requests": measured,
    }
