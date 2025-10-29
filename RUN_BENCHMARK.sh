#!/bin/bash
# Benchmark KV Cache Offloading Workflow
# This script generates a workload and runs benchmarks on both PCIe and C2C systems

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - Adjust these for your setup
# ============================================================================

# Model to use (must be compatible with TensorRT-LLM)
MODEL="meta-llama/Llama-3-70B-Instruct"  # Change to your model path/ID

# Workload parameters
CONTEXTS="32768 65536 131072"            # Context sizes in tokens
BATCHES_PER_CONTEXT=5                     # Number of batches per context size
BATCH_SIZES="8 16 32"                     # Possible batch sizes
PROMPT_POOL_SIZE=16                       # Distinct prompts per context (will be reused)
REUSE_START_BATCH=3                       # Start reusing prompts after this batch

# File paths
WORKLOAD_JSON="./workload.json"
RESULTS_JSON="./benchmark_results.json"

# Optional: Notes for this run
PCIe_NOTES="PCIe Gen5 baseline - H100"
C2C_NOTES="C2C coherent link - GH200"

# ============================================================================
# STEP 1: Generate Workload
# ============================================================================

echo "=========================================="
echo "Step 1: Generating workload..."
echo "=========================================="

python3 workload_generator.py \
    --contexts $CONTEXTS \
    --batches-per-context $BATCHES_PER_CONTEXT \
    --batch-sizes $BATCH_SIZES \
    --prompt-pool-size $PROMPT_POOL_SIZE \
    --reuse-start-batch $REUSE_START_BATCH \
    --delay-ms 0 \
    --seed 42 \
    --out "$WORKLOAD_JSON"

echo "✓ Workload generated: $WORKLOAD_JSON"
echo ""

# ============================================================================
# STEP 2: Run Benchmark on PCIe System
# ============================================================================

echo "=========================================="
echo "Step 2: Running benchmark on PCIe system..."
echo "=========================================="
echo "NOTE: Run this command on your PCIe system (e.g., H100 + PCIe Gen5)"
echo ""

echo "python3 benchmark_kv_offload.py \\"
echo "    --model \"$MODEL\" \\"
echo "    --config_module config_pcie \\"
echo "    --system_label pcie \\"
echo "    --workload_json \"$WORKLOAD_JSON\" \\"
echo "    --output_json \"$RESULTS_JSON\" \\"
echo "    --max_batch_size 32 \\"
echo "    --max_seq_len 131072 \\"
echo "    --temperature 0.0 \\"
echo "    --top_p 1.0 \\"
echo "    --notes \"$PCIe_NOTES\""
echo ""

# Uncomment to actually run (or run manually on PCIe system):
# python3 benchmark_kv_offload.py \
#     --model "$MODEL" \
#     --config_module config_pcie \
#     --system_label pcie \
#     --workload_json "$WORKLOAD_JSON" \
#     --output_json "$RESULTS_JSON" \
#     --max_batch_size 32 \
#     --max_seq_len 131072 \
#     --temperature 0.0 \
#     --top_p 1.0 \
#     --notes "$PCIe_NOTES"

# ============================================================================
# STEP 3: Run Benchmark on C2C System
# ============================================================================

echo "=========================================="
echo "Step 3: Running benchmark on C2C system..."
echo "=========================================="
echo "NOTE: Run this command on your C2C system (e.g., GH200/GB200)"
echo "      Use the SAME workload.json and results.json files"
echo ""

echo "python3 benchmark_kv_offload.py \\"
echo "    --model \"$MODEL\" \\"
echo "    --config_module config_c2c \\"
echo "    --system_label c2c \\"
echo "    --workload_json \"$WORKLOAD_JSON\" \\"
echo "    --output_json \"$RESULTS_JSON\" \\"
echo "    --max_batch_size 32 \\"
echo "    --max_seq_len 131072 \\"
echo "    --temperature 0.0 \\"
echo "    --top_p 1.0 \\"
echo "    --notes \"$C2C_NOTES\""
echo ""

# Uncomment to actually run (or run manually on C2C system):
# python3 benchmark_kv_offload.py \
#     --model "$MODEL" \
#     --config_module config_c2c \
#     --system_label c2c \
#     --workload_json "$WORKLOAD_JSON" \
#     --output_json "$RESULTS_JSON" \
#     --max_batch_size 32 \
#     --max_seq_len 131072 \
#     --temperature 0.0 \
#     --top_p 1.0 \
#     --notes "$C2C_NOTES"

echo "=========================================="
echo "Workflow complete!"
echo "=========================================="
echo "Results are in: $RESULTS_JSON"
echo "Workload file: $WORKLOAD_JSON"
echo ""
echo "The results JSON contains both 'pcie' and 'c2c' runs for comparison."
