
"""
config_c2c.py — C2C (NVLink/C2C-style coherent link ≈900 GB/s effective)

Aggressive settings that assume fast, coherent CPU<->GPU memory:
- Large host_cache_size (encourage offload)
- Slightly larger tokens_per_block to improve transfer efficiency
- Small GPU max_tokens to force churn and showcase offload wins
Tweak via CLI passthrough flags if desired.
"""

from tensorrt_llm.llmapi import KvCacheConfig

# Defaults chosen to highlight offloading on C2C-class systems
DEFAULT_GPU_KV_MAX_TOKENS = 2048          # keep small to induce offload/reload churn
DEFAULT_TOKENS_PER_BLOCK   = 64           # bigger blocks improve throughput on fast links
DEFAULT_HOST_CACHE_GB      = 64           # give the host a big cache (adjust to your RAM)
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
    # - ~900 GB/s coherent path makes offload/reload faster than recompute for many sizes.
    # - Larger blocks reduce bookkeeping and increase effective bandwidth utilization.
    # - Large host cache makes multi-tenant reuse viable.
    return KvCacheConfig(
        enable_block_reuse=bool(enable_reuse),
        max_tokens=int(gpu_kv_max_tokens),
        tokens_per_block=int(tokens_per_block),
        host_cache_size=_gb_to_bytes(int(host_cache_gb)),
    )



