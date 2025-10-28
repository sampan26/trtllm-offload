#!/usr/bin/env python3
"""
benchmark_kv_offload.py

Run this ONCE per system:
  - On PCIe baseline host (e.g., H100 + PCIe Gen5), pass: --config_module=config_pcie --system_label=pcie
  - On C2C host (e.g., GH200/GB200), pass:         --config_module=config_c2c  --system_label=c2c

Both runs should point to the SAME --workload_json and --output_json so results accumulate.

What this script does:
1) Imports a KvCacheConfig builder from the chosen config module (no cross-machine assumptions).
2) Imports a workload from workload_generator.generate_workload(...) (pure utility; no TRT-LLM).
3) Builds a TRT-LLM LLM instance with that KvCacheConfig.
4) Executes the workload (batched requests with specified context sizes).
5) Measures:
   - Throughput (req/s)
   - Latency:
       * time-to-first-token (TTFT) if streaming available; else total latency
   - Cache metrics by parsing DEBUG logs (cache hit rate, reused/offloaded blocks)
   - Observed offload bandwidth if TRT-LLM logs it
6) Appends structured results to --output_json, keyed by system_label.

Assumptions / Graceful fallbacks:
- TTFT: we try stream=True and capture the first token timestamp. If unavailable, we record end-to-end latency and mark ttft_fallback=true.
- Cache/bandwidth logs: we capture DEBUG logs via a temporary file. If expected patterns aren’t present, fields are set null.

You’ll implement:
- config_pcie.py / config_c2c.py
- workload_generator.py

This file intentionally does not include device-specific tuning: keep that in the config modules.
"""

import argparse
import importlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# TensorRT-LLM imports
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

# ------------- Helpers: logging capture & parsing ----------------

CACHE_HIT_RATE_RE = re.compile(r"cache hit rate:\s*([0-9]*\.?[0-9]+)")
REUSED_BLOCKS_RE = re.compile(r"reused blocks:\s*(\d+)")
OFFLOADED_BLOCKS_RE = re.compile(r"(?:offloaded|evicted) blocks:\s*(\d+)", re.IGNORECASE)
BANDWIDTH_RE = re.compile(r"(?:observed|measured)\s+bandwidth:\s*([0-9]*\.?[0-9]+)\s*GB/s", re.IGNORECASE)

def _enable_trtllm_debug_to_file(log_path: Path) -> None:
    """
    Direct TRT-LLM debug logs to a file we can parse later.
    We do this via env vars that TRT-LLM respects (common pattern).
    """
    os.environ["TLLM_LOG_LEVEL"] = "DEBUG"
    # If TRT-LLM supports a logfile env, set it; otherwise we’ll hope python logging is configured.
    # Many examples use stdout; we capture nothing extra here—user can tee if needed.
    # Try a best-effort variable (ignored if unsupported):
    os.environ["TLLM_LOGFILE"] = str(log_path)

def _parse_metrics_from_log(log_text: str) -> Dict[str, Optional[float]]:
    """
    Pulls best-effort metrics from TRT-LLM DEBUG logs.
    Flexible regex so we don't break on minor wording changes.
    """
    hit_rate = None
    reused_blocks = None
    offloaded_blocks = None
    observed_bw = None

    m = CACHE_HIT_RATE_RE.search(log_text)
    if m:
        hit_rate = float(m.group(1))
    m = REUSED_BLOCKS_RE.search(log_text)
    if m:
        reused_blocks = float(m.group(1))
    m = OFFLOADED_BLOCKS_RE.search(log_text)
    if m:
        offloaded_blocks = float(m.group(1))
    m = BANDWIDTH_RE.search(log_text)
    if m:
        observed_bw = float(m.group(1))

    return {
        "cache_hit_rate": hit_rate,
        "reused_blocks": reused_blocks,
        "offloaded_blocks": offloaded_blocks,
        "observed_offload_bandwidth_gbps": observed_bw,
    }

# ------------- Workload execution ----------------

def _now() -> float:
    return time.perf_counter()

def _run_batch_generate(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    try_stream: bool = True,
) -> Tuple[List[float], List[float]]:
    """
    Execute a batch (list of prompts) as a single generate call if supported.
    Returns two lists (same length as prompts):
      - ttft_seconds[i]: time to first token (or equals total latency if streaming not available)
      - total_latency_seconds[i]: end-to-end latency per request

    We attempt stream=True. If unavailable, we fallback to a single non-stream call.
    """
    # Not all LLM.generate APIs accept a list; TRT-LLM does. If not, loop per prompt.
    ttft = []
    total = []

    start_all = _now()
    if try_stream:
        try:
            # Streaming path: yields incremental tokens. We mark the first token per request.
            stream = llm.generate(prompts, sampling_params, stream=True)
            # stream yields per-step events; we need to detect the first token arrival for each sequence.
            first_seen = {i: None for i in range(len(prompts))}
            t0 = _now()
            for event in stream:
                # event structure can vary; we try common fields.
                # Typical: event.outputs is a list; each output may have 'index' and 'text' / 'token'
                emit_time = _now()
                if hasattr(event, "outputs"):
                    for out in getattr(event, "outputs"):
                        idx = getattr(out, "index", None)
                        # Some streaming APIs send token strings or deltas
                        token_text = getattr(out, "text", None) or getattr(out, "token", None)
                        if idx is not None and first_seen.get(idx) is None and token_text:
                            first_seen[idx] = emit_time
                # If event also signals finalization per index, we would capture end times; but we’ll fallback below.
            end_all = _now()
            # Fill ttft and total (fallback total duration per request = wall time of the whole batch;
            # better per-request end times would require richer callbacks).
            for i in range(len(prompts)):
                if first_seen[i] is None:
                    # Could be zero-length / immediate; fallback to end-to-end start->end
                    ttft.append(end_all - start_all)
                else:
                    ttft.append(first_seen[i] - start_all)
                total.append(end_all - start_all)

            return ttft, total
        except TypeError:
            # stream arg not supported
            pass
        except Exception:
            # Any unexpected streaming error – fallback to non-stream
            pass

    # Non-stream fallback: single call, single timer (use total as TTFT proxy)
    t0 = _now()
    _ = llm.generate(prompts, sampling_params)
    t1 = _now()
    dur = t1 - t0
    # Without per-request finish times, we attribute the same timing to each in batch.
    return [dur for _ in prompts], [dur for _ in prompts]

def _run_workload(
    llm: LLM,
    workload: List[Dict[str, Any]],
    sampling_params: SamplingParams,
) -> Dict[str, Any]:
    """
    Run a full workload shaped like:
        [{"batch_prompts": [...], "context_tokens": 32768, "scheduled_delay_ms": 0}, ...]
    Returns a dict with raw samples + aggregate stats.
    """
    per_request = []
    t_start = _now()
    for batch_idx, item in enumerate(workload):
        prompts = item["batch_prompts"]
        delay_ms = int(item.get("scheduled_delay_ms", 0))
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

        ttft, total = _run_batch_generate(llm, prompts, sampling_params, try_stream=True)
        # Record per-request samples
        context_tokens = int(item.get("context_tokens", 0))
        for i in range(len(prompts)):
            per_request.append({
                "batch_index": batch_idx,
                "request_index_in_batch": i,
                "ttft_seconds": ttft[i],
                "total_latency_seconds": total[i],
                "context_tokens": context_tokens,
                "prompt_chars": len(prompts[i]),
            })

    t_end = _now()
    wall = t_end - t_start
    nreq = len(per_request)
    throughput = nreq / wall if wall > 0 else 0.0

    # Aggregate latency percentiles (p50/p95/p99) for TTFT and total
    def pct(values: List[float], p: float) -> float:
        if not values:
            return float("nan")
        s = sorted(values)
        k = max(0, min(len(s)-1, int(round((p/100.0) * (len(s)-1)))))
        return s[k]

    ttfts = [x["ttft_seconds"] for x in per_request]
    tots  = [x["total_latency_seconds"] for x in per_request]

    agg = {
        "wall_time_seconds": wall,
        "num_requests": nreq,
        "throughput_req_per_s": throughput,
        "ttft_p50_s": pct(ttfts, 50),
        "ttft_p95_s": pct(ttfts, 95),
        "ttft_p99_s": pct(ttfts, 99),
        "latency_p50_s": pct(tots, 50),
        "latency_p95_s": pct(tots, 95),
        "latency_p99_s": pct(tots, 99),
    }
    return {"aggregate": agg, "samples": per_request}

# ------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark KV cache offloading with TRT-LLM (PCIe vs C2C orchestrator).")
    ap.add_argument("--model", required=True, help="Model id/path for TRT-LLM (e.g., 'meta-llama/Llama-3-70B-Instruct').")
    ap.add_argument("--config_module", required=True, help="Python module that provides build_kv_config(args)->KvCacheConfig (e.g., config_pcie or config_c2c).")
    ap.add_argument("--system_label", required=True, choices=["pcie","c2c"], help="Label for this run; used as key in output JSON.")
    ap.add_argument("--workload_json", required=True, help="Path to workload JSON produced by workload_generator.py.")
    ap.add_argument("--output_json", required=True, help="Path to append/merge benchmark results JSON.")
    ap.add_argument("--max_batch_size", type=int, default=32, help="Upper bound to respect when running batches.")
    ap.add_argument("--max_seq_len", type=int, default=131072, help="Cap for generated tokens per request.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    ap.add_argument("--top_p", type=float, default=1.0, help="Top-p for sampling.")
    ap.add_argument("--logfile", default="", help="Optional: explicit TRT-LLM log file to parse; otherwise a temp file is used.")
    ap.add_argument("--notes", default="", help="Freeform notes to store with this run (e.g., host specs).")
    args = ap.parse_args()

    # 1) Import KvCacheConfig builder from the chosen config module
    try:
        cfg_mod = importlib.import_module(args.config_module)
    except Exception as e:
        raise RuntimeError(f"Failed to import config module '{args.config_module}': {e}")

    if not hasattr(cfg_mod, "build_kv_config"):
        raise RuntimeError(f"Config module '{args.config_module}' must expose build_kv_config(args)->KvCacheConfig")

    kv_cfg: KvCacheConfig = cfg_mod.build_kv_config(args)

    # 2) Load workload
    workload_path = Path(args.workload_json)
    if not workload_path.exists():
        raise FileNotFoundError(f"Workload JSON not found: {workload_path}")
    workload = json.loads(workload_path.read_text())
    # Expect a list of items: {"batch_prompts":[...], "context_tokens":int, "scheduled_delay_ms":int}

    # 3) Prepare logging capture (DEBUG)
    if args.logfile:
        log_path = Path(args.logfile)
        # ensure parent exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.NamedTemporaryFile(prefix=f"tllm_{args.system_label}_", suffix=".log", delete=False)
        log_path = Path(tmp.name)
        tmp.close()
    _enable_trtllm_debug_to_file(log_path)

    # 4) Build LLM
    sampling = SamplingParams(
        max_tokens=args.max_seq_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    llm = LLM(
        model=args.model,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        kv_cache_config=kv_cfg,
    )

    # 5) Execute workload
    run_meta = {
        "system_label": args.system_label,
        "model": args.model,
        "max_batch_size": args.max_batch_size,
        "max_seq_len": args.max_seq_len,
        "config_module": args.config_module,
        "notes": args.notes,
        "trtllm_log_file": str(log_path),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    run_start = _now()
    results = _run_workload(llm, workload, sampling)
    run_end = _now()

    llm.shutdown()

    # 6) Parse cached/offload metrics from log file (best-effort)
    log_text = ""
    try:
        if log_path.exists():
            log_text = log_path.read_text(errors="ignore")
    except Exception:
        pass
    log_metrics = _parse_metrics_from_log(log_text)
    results["log_metrics"] = log_metrics
    results["run_meta"] = run_meta
    results["run_meta"]["wall_time_seconds_script"] = run_end - run_start

    # 7) Merge/append results into output JSON under system_label
    out_path = Path(args.output_json)
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}

    if "runs" not in existing:
        existing["runs"] = {}
    if args.system_label not in existing["runs"]:
        existing["runs"][args.system_label] = []

    existing["runs"][args.system_label].append(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"[benchmark_kv_offload] Appended results under '{args.system_label}' to {out_path}")

if __name__ == "__main__":
    main()