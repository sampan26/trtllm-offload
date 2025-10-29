
"""
config_pcie.py — PCIe Gen5 baseline (≈64 GB/s host link)

Settings optimized for long-context serving with KV cache offloading:
- Moderate GPU cache that forces offload for long contexts (32K+ tokens)
- Larger block sizes to improve transfer efficiency on PCIe
- Sufficient host cache to enable offload/reload testing
You can override sizes at runtime with CLI flags passed through args.
"""

from tensorrt_llm.llmapi import KvCacheConfig

# Defaults for PCIe Gen5 (≈64 GB/s host link) with long-context offloading
# Note: Even on slower PCIe, we enable offloading to test reload behavior
DEFAULT_GPU_KV_MAX_TOKENS = 4096          # small GPU cache for long context (32K+ tokens) to force offload
DEFAULT_TOKENS_PER_BLOCK   = 64           # larger blocks improve transfer efficiency even on PCIe
DEFAULT_HOST_CACHE_GB      = 32           # enable offloading to test reload from host memory
DEFAULT_ENABLE_REUSE       = True         # enable block reuse for offload/reload testing

def _gb_to_bytes(gb: int) -> int:
    return int(gb) * (1024 ** 3)

def build_kv_config(args) -> KvCacheConfig:
    """
    Optional passthrough CLI flags (add these to benchmark_kv_offload.py if you want):
      --pcie_gpu_kv_max_tokens   (int)
      --pcie_tokens_per_block    (int)
      --pcie_host_cache_gb       (int)
      --pcie_enable_reuse        (bool via presence/absence; see your argparse)
    If not present, we use the defaults above.
    """
    gpu_kv_max_tokens = getattr(args, "pcie_gpu_kv_max_tokens", DEFAULT_GPU_KV_MAX_TOKENS)
    tokens_per_block  = getattr(args, "pcie_tokens_per_block",  DEFAULT_TOKENS_PER_BLOCK)
    host_cache_gb     = getattr(args, "pcie_host_cache_gb",     DEFAULT_HOST_CACHE_GB)
    enable_reuse      = getattr(args, "pcie_enable_reuse",      DEFAULT_ENABLE_REUSE)

    # Rationale:
    # - 64 GB/s PCIe is slower than C2C, but offloading still beneficial for long contexts vs recompute
    # - Larger blocks (64 tokens) improve transfer efficiency; overhead amortized over long contexts
    # - 32GB host cache sufficient to test offload/reload behavior with reasonable capacity
    # - 4096 token GPU cache allows initial caching, then forces offload for long contexts (32K+)
    return KvCacheConfig(
        enable_block_reuse=bool(enable_reuse),
        max_tokens=int(gpu_kv_max_tokens),
        tokens_per_block=int(tokens_per_block),
        host_cache_size=_gb_to_bytes(int(host_cache_gb)),
    )
