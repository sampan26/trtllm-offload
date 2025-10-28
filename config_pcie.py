
"""
config_pcie.py — PCIe Gen5 baseline (≈64 GB/s host link)

Conservative settings that reflect the high cost of host<->device transfers:
- Small/zero host_cache_size (effectively discourages offload)
- Modest tokens_per_block to limit transfer granularity
- Small max_tokens on-GPU to induce eviction/recompute on PCIe
You can override sizes at runtime with CLI flags passed through args.
"""

from tensorrt_llm.llmapi import KvCacheConfig

# Defaults chosen to bias against offloading on PCIe
DEFAULT_GPU_KV_MAX_TOKENS = 2048          # small GPU KV cache to create pressure
DEFAULT_TOKENS_PER_BLOCK   = 32           # modest block size -> less efficient transfers
DEFAULT_HOST_CACHE_GB      = 0            # disable offload by default on PCIe
DEFAULT_ENABLE_REUSE       = True         # still allow block reuse if it happens

def _gb_to_bytes(gb: int) -> int:
    return int(gb) * (1024 ** 3)

def build_kv_config(args) -> KvCacheConfig:
    """
    Optional passthrough CLI flags (add these to benchmark_kv_offload.py if you want):
      --pcie_gpu_kv_max_tokens   (int)
      --pcie_tokens_per_block    (int)
      --pcie_host_cache_gb       (int)
      --pcie_enable_reuse        (bool via presence/absence; see your argparse)
    If not present, we use the conservative defaults above.
    """
    gpu_kv_max_tokens = getattr(args, "pcie_gpu_kv_max_tokens", DEFAULT_GPU_KV_MAX_TOKENS)
    tokens_per_block  = getattr(args, "pcie_tokens_per_block",  DEFAULT_TOKENS_PER_BLOCK)
    host_cache_gb     = getattr(args, "pcie_host_cache_gb",     DEFAULT_HOST_CACHE_GB)
    enable_reuse      = getattr(args, "pcie_enable_reuse",      DEFAULT_ENABLE_REUSE)

    # Rationale:
    # - 64 GB/s PCIe => large offloads can be slower than recompute; keep host cache tiny/disabled.
    # - Smaller blocks reduce single-transfer size but add overhead; that’s realistic for PCIe costs.
    return KvCacheConfig(
        enable_block_reuse=bool(enable_reuse),
        max_tokens=int(gpu_kv_max_tokens),
        tokens_per_block=int(tokens_per_block),
        host_cache_size=_gb_to_bytes(int(host_cache_gb)),
    )