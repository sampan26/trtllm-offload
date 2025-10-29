
"""
config_c2c.py — C2C (NVLink/C2C-style coherent link ≈900 GB/s effective)

Aggressive settings optimized for long-context serving with fast offloading:
- Moderate GPU cache that forces offload for long contexts (32K+ tokens)
- Very large block sizes to maximize bandwidth on fast coherent links
- Large host cache for multi-tenant long-context serving
Tweak via CLI passthrough flags if desired.
"""

from tensorrt_llm.llmapi import KvCacheConfig

# Defaults for C2C (NVLink/C2C-style coherent link ≈900 GB/s effective) with long-context offloading
DEFAULT_GPU_KV_MAX_TOKENS = 8192          # small GPU cache relative to long contexts (32K+ tokens) to force offload
DEFAULT_TOKENS_PER_BLOCK   = 128          # large blocks maximize throughput on fast coherent links
DEFAULT_HOST_CACHE_GB      = 128          # large host cache for multi-tenant long-context serving
DEFAULT_ENABLE_REUSE       = True

def _gb_to_bytes(gb: int) -> int:
    return int(gb) * (1024 ** 3)

def build_kv_config(args) -> KvCacheConfig:
    """
    Optional passthrough CLI flags (add to your argparse if you want):
      --c2c_gpu_kv_max_tokens  (int)
      --c2c_tokens_per_block   (int)
      --c2c_host_cache_gb      (int)
      --c2c_enable_reuse       (bool)
    """
    gpu_kv_max_tokens = getattr(args, "c2c_gpu_kv_max_tokens", DEFAULT_GPU_KV_MAX_TOKENS)
    tokens_per_block  = getattr(args, "c2c_tokens_per_block",  DEFAULT_TOKENS_PER_BLOCK)
    host_cache_gb     = getattr(args, "c2c_host_cache_gb",     DEFAULT_HOST_CACHE_GB)
    enable_reuse      = getattr(args, "c2c_enable_reuse",      DEFAULT_ENABLE_REUSE)

    # Rationale:
    # - ~900 GB/s coherent path makes offload/reload much faster than recompute for long contexts
    # - Very large blocks (128 tokens) maximize bandwidth utilization on fast links
    # - 128GB host cache supports multiple concurrent long-context sessions
    # - 8192 token GPU cache allows initial prefetch, then offloads remainder of long contexts (32K+)
    return KvCacheConfig(
        enable_block_reuse=bool(enable_reuse),
        max_tokens=int(gpu_kv_max_tokens),
        tokens_per_block=int(tokens_per_block),
        host_cache_size=_gb_to_bytes(int(host_cache_gb)),
    )


