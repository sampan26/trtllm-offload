# Benchmark KV Cache Offloading - Command Reference

This document provides step-by-step commands to run the full benchmarking workflow.

## Prerequisites

- Python 3 with TensorRT-LLM installed
- `tiktoken` (optional, for accurate token counting)
- Access to both PCIe and C2C systems (or run sequentially on same system with different configs)

## Step 1: Generate Workload

Generate a workload JSON file once. This file will be used by both PCIe and C2C benchmarks.

```bash
python3 workload_generator.py \
    --contexts 32768 65536 131072 \
    --batches-per-context 5 \
    --batch-sizes 8 16 32 \
    --prompt-pool-size 16 \
    --reuse-start-batch 3 \
    --delay-ms 0 \
    --seed 42 \
    --out workload.json
```

**Parameters explained:**
- `--contexts`: Target context sizes in tokens (32K, 64K, 128K)
- `--batches-per-context`: Number of batches per context size
- `--batch-sizes`: Possible batch sizes to sample from
- `--prompt-pool-size`: Number of distinct prompts per context (will be reused later)
- `--reuse-start-batch`: After this batch, start reusing prompts to test offload/reload
- `--seed`: Random seed for reproducibility

## Step 2: Run Benchmark on PCIe System

Run this on your PCIe system (e.g., H100 + PCIe Gen5).

```bash
python3 benchmark_kv_offload.py \
    --model "meta-llama/Llama-3-70B-Instruct" \
    --config_module config_pcie \
    --system_label pcie \
    --workload_json workload.json \
    --output_json benchmark_results.json \
    --max_batch_size 32 \
    --max_seq_len 131072 \
    --temperature 0.0 \
    --top_p 1.0 \
    --notes "PCIe Gen5 baseline - H100"
```

**Parameters explained:**
- `--model`: Your TRT-LLM model path/ID
- `--config_module`: Use `config_pcie` for PCIe configuration
- `--system_label`: Must be `pcie` or `c2c`
- `--workload_json`: Path to workload generated in Step 1
- `--output_json`: Results file (will accumulate both runs)
- `--max_batch_size`: Maximum batch size for inference
- `--max_seq_len`: Maximum sequence length (should match your longest context)
- `--notes`: Freeform notes about this run

## Step 3: Run Benchmark on C2C System

Run this on your C2C system (e.g., GH200/GB200). **Use the SAME workload.json and results.json files.**

```bash
python3 benchmark_kv_offload.py \
    --model "meta-llama/Llama-3-70B-Instruct" \
    --config_module config_c2c \
    --system_label c2c \
    --workload_json workload.json \
    --output_json benchmark_results.json \
    --max_batch_size 32 \
    --max_seq_len 131072 \
    --temperature 0.0 \
    --top_p 1.0 \
    --notes "C2C coherent link - GH200"
```

**Important:** The `--workload_json` and `--output_json` must be the same files used in Step 2. The results will be merged into the same JSON file under separate keys (`pcie` and `c2c`).

## Optional: Override Config Parameters

You can override the default config values via CLI arguments. For example, to use a larger GPU cache on PCIe:

```bash
python3 benchmark_kv_offload.py \
    --model "meta-llama/Llama-3-70B-Instruct" \
    --config_module config_pcie \
    --system_label pcie \
    --workload_json workload.json \
    --output_json benchmark_results.json \
    --pcie_gpu_kv_max_tokens 8192 \
    --pcie_host_cache_gb 64 \
    --notes "PCIe with larger cache"
```

Available overrides:
- **PCIe:** `--pcie_gpu_kv_max_tokens`, `--pcie_tokens_per_block`, `--pcie_host_cache_gb`, `--pcie_enable_reuse`
- **C2C:** `--c2c_gpu_kv_max_tokens`, `--c2c_tokens_per_block`, `--c2c_host_cache_gb`, `--c2c_enable_reuse`

## Output Format

The `benchmark_results.json` file will contain:

```json
{
  "runs": {
    "pcie": [
      {
        "aggregate": {
          "throughput_req_per_s": 2.5,
          "ttft_p50_s": 0.123,
          "ttft_p95_s": 0.456,
          "ttft_p99_s": 0.789,
          "latency_p50_s": 1.234,
          "latency_p95_s": 2.345,
          "latency_p99_s": 3.456
        },
        "log_metrics": {
          "cache_hit_rate": 0.85,
          "reused_blocks": 1024,
          "offloaded_blocks": 2048,
          "observed_offload_bandwidth_gbps": 45.2
        },
        "samples": [...],
        "run_meta": {...}
      }
    ],
    "c2c": [...]
  }
}
```

## Quick Reference: Minimal Commands

If you want to get started quickly with defaults:

```bash
# 1. Generate workload
python3 workload_generator.py --contexts 32768 65536 --out workload.json

# 2. Run on PCIe (on PCIe system)
python3 benchmark_kv_offload.py \
    --model "your-model-path" \
    --config_module config_pcie \
    --system_label pcie \
    --workload_json workload.json \
    --output_json results.json

# 3. Run on C2C (on C2C system, same files)
python3 benchmark_kv_offload.py \
    --model "your-model-path" \
    --config_module config_c2c \
    --system_label c2c \
    --workload_json workload.json \
    --output_json results.json
```

## Troubleshooting

1. **Model not found**: Make sure your model path is correct and TensorRT-LLM can load it
2. **Workload file not found**: Ensure you generated the workload.json file first
3. **Import errors**: Make sure `config_pcie.py` and `config_c2c.py` are in the same directory
4. **Out of memory**: Reduce `--max_batch_size` or `--prompt-pool-size` in workload generation
