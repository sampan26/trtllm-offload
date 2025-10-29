#!/usr/bin/env python3
"""
workload_generator.py — Pure utility to build a reproducible long-context workload.

It produces a JSON list where each item looks like:
  {
    "batch_prompts": ["<prompt1>", ..., "<promptN>"],
    "context_tokens": 65536,
    "scheduled_delay_ms": 0
  }

Key ideas:
- You choose target context lengths (e.g., 32K, 64K, 128K tokens).
- Simulates serving systems where:
  1. Initial requests come in → KV cache computed and stored on GPU
  2. More requests arrive → GPU cache full → earlier KV caches offloaded to CPU/host
  3. Original requests return → KV cache reloaded from offloaded memory (not recomputed)
- Creates a pool of distinct prompts, then reuses exact duplicates later in the workload
- No TRT-LLM dependency. Optional tokenizer: if `tiktoken` is installed, we'll use it for
  token-accurate sizing; otherwise we approximate 4 chars/token.

Usage example:
  python workload_generator.py \
    --contexts 32768 65536 131072 \
    --batches-per-context 5 \
    --batch-sizes 8 16 32 \
    --prompt-pool-size 16 \
    --reuse-start-batch 3 \
    --delay-ms 0 \
    --seed 42 \
    --out workload.json
"""

import argparse
import json
import math
import random
import string
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import tiktoken  # optional
except Exception:
    tiktoken = None

AVG_CHARS_PER_TOKEN = 4  # crude fallback when tokenizer isn't available

def _token_count(s: str) -> int:
    if tiktoken is None:
        return max(1, len(s) // AVG_CHARS_PER_TOKEN)
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(s))

def _make_paragraphs(n: int, seed: int) -> List[str]:
    """
    Build a deterministic pool of 'n' paragraphs (pseudo-doc/rag snippets).
    We mix dictionary-like tokens, code-ish lines, and prose to defeat trivial dedupe.
    """
    rng = random.Random(seed)
    paras = []
    vocab_words = [
        "vectorized", "scheduling", "allocator", "latency", "bandwidth", "throughput",
        "attention", "kv-cache", "tokenizer", "prefill", "decode", "pipeline", "coalesce",
        "fragmentation", "reuse", "paging", "numa", "coherent", "rendezvous", "eviction",
        "recompute", "speculative", "quantization", "tensor", "kernel", "warp", "stream"
    ]
    for i in range(n):
        w = rng.sample(vocab_words, k=min(len(vocab_words), 8))
        # add a code-ish stanza and a semi-random sentence
        code = f"// block={rng.randint(8,128)} tokens; stride={rng.choice([16,32,64])}; reuse={rng.choice([0,1])}\n"
        sent = " ".join([
            rng.choice(w).capitalize(),
            "paths favor",
            rng.choice(["coherent", "pooled", "paged"]),
            "memory under bursty multi-tenant loads."
        ])
        filler = "".join(rng.choices(string.ascii_lowercase + " ", k=200))
        paras.append(f"{sent}\n{code}{filler}\n")
    return paras

def _build_prompt(target_tokens: int, base_paragraphs: List[str]) -> str:
    """
    Concatenate paragraphs in a round-robin fashion until we hit ~target_tokens.
    If we overshoot, we trim to the closest boundary.
    """
    buf: List[str] = []
    t = 0
    i = 0
    n = len(base_paragraphs)
    # guard against empty
    if n == 0:
        base_paragraphs = ["lorem ipsum " * 100]
        n = 1

    while t < target_tokens:
        buf.append(base_paragraphs[i % n])
        i += 1
        t = _token_count("".join(buf))

        # If we overshoot heavily, try trimming the last paragraph roughly
        if t > target_tokens and i > 0:
            last = base_paragraphs[(i - 1) % n]
            # binary chop down the last paragraph if tokenizer is available, else crude slice
            lo, hi = 0, len(last)
            best = last
            if tiktoken:
                for _ in range(12):
                    mid = (lo + hi) // 2
                    cand = "".join(buf[:-1]) + last[:mid]
                    tc = _token_count(cand)
                    if tc >= target_tokens:
                        best = last[:mid]
                        hi = mid
                    else:
                        lo = mid + 1
                buf[-1] = best
            else:
                # approximate chars per token
                over = t - target_tokens
                cut_chars = max(0, over * AVG_CHARS_PER_TOKEN)
                buf[-1] = last[:-cut_chars] if cut_chars < len(last) else ""
            break

    return "".join(buf)

def generate_workload(
    contexts: List[int],
    batches_per_context: int,
    batch_sizes: List[int],
    prompt_pool_size: int,
    reuse_start_batch: int,
    delay_ms: int,
    seed: int,
) -> List[dict]:
    """
    Generate workload that simulates serving systems with KV cache offloading.
    
    Args:
        contexts: Target context sizes in tokens
        batches_per_context: Number of batches per context size
        batch_sizes: List of possible batch sizes to sample from
        prompt_pool_size: Number of distinct prompts to create per context size
        reuse_start_batch: After this many batches, start reusing prompts from the pool
                         (simulates original requests returning after being offloaded)
        delay_ms: Inter-batch delay
        seed: Random seed
    """
    rng = random.Random(seed)
    items = []
    
    for ctx in contexts:
        # Create a pool of distinct full prompts for this context size
        # These will be reused later to test offload/reload behavior
        prompt_pool = []
        for pool_idx in range(prompt_pool_size):
            # Create unique prompts by varying the seed per pool index
            paras = _make_paragraphs(n=16, seed=seed + ctx + 1000 * pool_idx)
            base_prompt = _build_prompt(ctx, paras)
            # Add a unique user query to make it a complete request
            user_query = f"\n\nUSER: Please summarize the preceding context in 2 bullets. Request ID: {pool_idx}"
            full_prompt = base_prompt + user_query
            prompt_pool.append(full_prompt)
        
        batch_idx = 0
        for _ in range(batches_per_context):
            bs = rng.choice(batch_sizes)
            batch_prompts = []
            
            if batch_idx < reuse_start_batch:
                # Initial phase: use prompts from pool (will fill GPU cache)
                # After GPU cache fills, these will be offloaded
                for i in range(bs):
                    prompt_idx = (batch_idx * max(batch_sizes) + i) % len(prompt_pool)
                    batch_prompts.append(prompt_pool[prompt_idx])
            else:
                # Reuse phase: reintroduce exact same prompts to test reload from offloaded memory
                # Mix some reused prompts with some new ones
                reuse_ratio = 0.7  # 70% reused, 30% new
                for i in range(bs):
                    if rng.random() < reuse_ratio:
                        # Reuse a prompt from the pool (exact duplicate)
                        prompt_idx = rng.randint(0, len(prompt_pool) - 1)
                        batch_prompts.append(prompt_pool[prompt_idx])
                    else:
                        # Occasionally add a new prompt to maintain some churn
                        new_pool_idx = len(prompt_pool) + (batch_idx * max(batch_sizes) + i)
                        paras = _make_paragraphs(n=16, seed=seed + ctx + 1000 * new_pool_idx)
                        base_prompt = _build_prompt(ctx, paras)
                        user_query = f"\n\nUSER: Please summarize the preceding context in 2 bullets. Request ID: {new_pool_idx}"
                        batch_prompts.append(base_prompt + user_query)
            
            items.append({
                "batch_prompts": batch_prompts,
                "context_tokens": ctx,
                "scheduled_delay_ms": delay_ms
            })
            batch_idx += 1
    
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", required=True,
                    help="Target context sizes in tokens, e.g., 32768 65536 131072")
    ap.add_argument("--batches-per-context", type=int, default=4,
                    help="Number of batches per context size")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32],
                    help="Possible batch sizes to sample from")
    ap.add_argument("--prompt-pool-size", type=int, default=16,
                    help="Number of distinct prompts to create per context size (will be reused)")
    ap.add_argument("--reuse-start-batch", type=int, default=3,
                    help="After this many batches, start reusing prompts from pool (tests offload/reload)")
    ap.add_argument("--delay-ms", type=int, default=0, help="Inter-batch delay to simulate bursts.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    wl = generate_workload(
        contexts=args.contexts,
        batches_per_context=args.batches_per_context,
        batch_sizes=args.batch_sizes,
        prompt_pool_size=args.prompt_pool_size,
        reuse_start_batch=args.reuse_start_batch,
        delay_ms=args.delay_ms,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(wl, indent=2))
    print(f"[workload_generator] wrote {len(wl)} batches to {args.out}")

if __name__ == "__main__":
    main()
