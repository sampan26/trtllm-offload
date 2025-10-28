
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
- For each context length and requested batch sizes, we synthesize tenant-style prompts
  (A, B, C, ...) by concatenating semi-distinct paragraphs to approximate the token length.
- No TRT-LLM dependency. Optional tokenizer: if `tiktoken` is installed, we’ll use it for
  token-accurate sizing; otherwise we approximate 4 chars/token.

Usage example:
  python workload_generator.py \
    --contexts 32768 65536 131072 \
    --batches-per-context 5 \
    --batch-sizes 8 16 32 \
    --tenants 4 \
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

def _tenant_prompts(num_tenants: int, target_tokens: int, seed: int) -> List[str]:
    """
    Build 'num_tenants' distinct long system prompts of ~target_tokens each.
    """
    # Each tenant gets its own paragraph pool seed to keep overlap limited
    prompts = []
    for k in range(num_tenants):
        paras = _make_paragraphs(n=16, seed=seed + 97 * (k + 1))
        p = _build_prompt(target_tokens, paras)
        prompts.append(p)
    return prompts

def generate_workload(
    contexts: List[int],
    batches_per_context: int,
    batch_sizes: List[int],
    tenants: int,
    delay_ms: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    items = []
    # Prebuild tenants per context size for realism
    for ctx in contexts:
        tenant_system_prompts = _tenant_prompts(tenants, ctx, seed=seed + ctx)
        for _ in range(batches_per_context):
            bs = rng.choice(batch_sizes)
            batch_prompts = []
            # Alternate tenants to induce churn
            for i in range(bs):
                t_idx = i % tenants
                # Short user turn appended to each system prompt to avoid exact dedupe
                user = f"\n\nUSER[{i}]: Please summarize the preceding context in 2 bullets."
                batch_prompts.append(tenant_system_prompts[t_idx] + user)
            items.append({
                "batch_prompts": batch_prompts,
                "context_tokens": ctx,
                "scheduled_delay_ms": delay_ms
            })
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", required=True,
                    help="Target context sizes in tokens, e.g., 32768 65536 131072")
    ap.add_argument("--batches-per-context", type=int, default=4)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--tenants", type=int, default=4, help="Distinct long system prompts per context.")
    ap.add_argument("--delay-ms", type=int, default=0, help="Inter-batch delay to simulate bursts.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    wl = generate_workload(
        contexts=args.contexts,
        batches_per_context=args.batches_per_context,
        batch_sizes=args.batch_sizes,
        tenants=args.tenants,
        delay_ms=args.delay_ms,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(wl, indent=2))
    print(f"[workload_generator] wrote {len(wl)} batches to {args.out}")

if __name__ == "__main__":
    main()
