#!/bin/bash
set -e

# ============================================================
# vLLM Server Startup Script - Non-Thinking Mode (Config D)
# ============================================================
# Model:      Qwen3.6-27B-FP8
# Hardware:   2x AMD Radeon RX 9700 PRO (gfx1201/RDNA4)
# Optimized:  BF16 KV cache + MTP speculative decoding
# Thinking:   OFF (enable_thinking: false)
# Port:       8000
#
# Key design decisions:
#   - No FP8 KV cache: degrades decode 4-7x on RDNA4 (benchmarked)
#   - MTP speculative decoding: +94-148% generation speed
#   - TRITON_ATTN: stable, avoids bimodal decode bug (ROCm#6347)
#   - No chunked prefill: incompatible with MTP
# ============================================================

# Navigate to the environment root so uv automatically detects the .venv
cd /AITOOLCHAIN/vllm-server

# --- Safety override for 262k massive contexts ---
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# --- Library paths ---
# Inject isolated OpenMPI 4 libraries ahead of the system's OpenMPI 5 stack
export LD_LIBRARY_PATH="/AITOOLCHAIN/vllm-server/ompi4-compat:$LD_LIBRARY_PATH"

# --- ROCm bare-metal optimizations ---
# Accelerate loading safetensors directly into VRAM
export SAFETENSORS_FAST_GPU=1
# Force device kernel arguments for faster launch times
export HIP_FORCE_DEV_KERNARG=1
# Prevent ROCm IPC fork deadlocks on multi-GPU setups
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Ensure we only bind to the two target GPUs (adjust if you have an integrated GPU)
export HIP_VISIBLE_DEVICES=0,1

# --- RDNA4 stability fixes (from kyuz0/amd-r9700-vllm-toolboxes) ---
# Fix TP=2 RCCL deadlock on RDNA4 (ROCm/rocm-systems#5480)
export NCCL_PROTO=Simple
# Reduce GPU memory fragmentation at 262K context
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- Optional: tcmalloc for clean shutdown ---
# Fixes double-free crash on vLLM shutdown. Uncomment if the library exists.
# Verify path with: find /usr -name "libtcmalloc_minimal*" 2>/dev/null
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

echo "================================================================"
echo "  vLLM Server - Non-Thinking Mode (Config D)"
echo "  Model:     Qwen3.6-27B-FP8"
echo "  Hardware:  2x RX 9700 PRO (TP=2)"
echo "  KV Cache:  BF16 (no FP8 - fixes RDNA4 decode degradation)"
echo "  Spec Dec:  MTP (2 speculative tokens)"
echo "  Thinking:  OFF"
echo "  Port:      8000"
echo "================================================================"

# --- Launch vLLM ---
uv run vllm serve /AITOOLCHAIN/models/Qwen3.6-27B-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --attention-backend TRITON_ATTN \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
  --compilation-config '{"pass_config":{"fuse_norm_quant":false}}' \
  --mm-encoder-attn-backend TRITON_ATTN \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --mm-shm-cache-max-object-size-mb 256
