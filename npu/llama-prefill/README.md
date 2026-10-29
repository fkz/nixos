# Experimental llama.cpp XDNA2 prefill bridge

This patch adds an opt-in NPU path inside the Vulkan backend of llama.cpp 0.8.0.
It supports native IQ3_XXS MoE expert gate/up matrices with K=2560 and N=640,
including the matching tensors in ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF.
Other formats, shapes, and small expert groups use Vulkan. Single-token decode
stays on Vulkan. This is partial prefill acceleration, not an all-NPU runtime.

The NPU decodes original packed weights and performs INT8 matrix multiplication
with BF16 partial accumulation. This adds approximation beyond the GGUF format.
The standalone matrix test measured about 1.9% relative L2 tensor error against
FP32 multiplication of the original decoded weights. Full-model quality has not been evaluated. A 128-token full-model smoke
benchmark measured 34.2 tokens/s on Vulkan versus 25.4 tokens/s when every
matching expert group was forced onto the NPU (threshold 1). This proves the
path executes, but forcing small expert groups currently reduces throughput.
A 512-token test with the default threshold measured 61.0 tokens/s versus
60.1 on Vulkan (two repetitions, within run-to-run variation). Logs confirmed
actual hybrid NPU/Vulkan routing; this does not establish a reliable speedup.

## Run

Without a system activation:

```sh
GGML_VK_FORCE_MMVQ=1 nix run /home/fabian/nixos#llama-npu -- serve \
  -hf ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_XXS --lazy-mode on
```

After activating with `sudo nixos-update`, use `llama-npu serve` with the same
arguments. Ordinary `llama` keeps the NPU disabled unless explicitly enabled
with `GGML_VK_NPU_PREFILL=1`.

The package wrapper supplies the matching XRT runtime/plugin and compiled
kernels. No driver patch or reboot is needed for this userspace integration.

## Scheduling and logs

`GGML_VK_NPU_MIN_EXPERT_TOKENS` defaults to 256. It counts routed rows for each
expert in the current microbatch, not total prompt tokens. Lower values can
force more NPU work for experiments but may reduce throughput. For a forced
execution check, prefix the command above with
`GGML_VK_NPU_MIN_EXPERT_TOKENS=1`; look for actual row counts in the logs. The default
favors larger expert batches, since the earlier standalone benchmark found
64-row batches slower than Vulkan.

Matching operations run at an explicit synchronization boundary. Eligible
expert groups execute on the NPU. Remaining token/expert pairs are rebatched
into a separate Vulkan MUL_MAT_ID without recomputing NPU rows. Results are
scattered back into their original token/slot positions before the graph
continues. GPU/NPU execution is serialized in this first integration.

Logs beginning `ggml_npu_prefill:` report actual NPU row and dispatch counts.
If the runtime cannot initialize or execute, it logs the error and disables
NPU offload for the rest of that process, falling back to Vulkan.

`GGML_VK_NPU_PACK_THREADS` controls CPU activation packing (default 6).
`GGML_VK_NPU_KERNEL_DIR` can override the installed kernel directory for tests.

## Hardware checks

`check-integration.py` exercises an operation before and after a mixed NPU/GPU
expert multiplication. It compares every NPU output with an independent
INT8/BF16 reference and every GPU remainder with ordinary Vulkan. It uses a
strided expert-ID view, matching the real model’s top-k layout, then changes
activations and weights to check for stale results. Modes `decode`, `fallback`,
and `missing` check single-token bypass, below-threshold batches, and missing
kernel fallback. The test can use deterministic synthetic weights, or a saved
native expert via `LLAMA_NPU_TEST_WEIGHTS`.

The test needs `LLAMA_TEST_ROOT` pointing at the patched llama package output,
`XILINX_XRT`, `LD_LIBRARY_PATH`, and `GGML_VK_NPU_KERNEL_DIR` set as in the package
wrapper, plus the NPU Python launcher. For `hybrid` mode, set
`GGML_VK_NPU_MIN_EXPERT_TOKENS=384`: one expert gets 512 rows on the NPU and two
others get 256 rows each on Vulkan. For `fallback`, set the threshold to 1024;
for `missing`, use a nonexistent kernel directory. Tests require real hardware
and are run outside the Nix build sandbox.

Kernel sources and references are adapted from phhusson/amd-npu-gguf commit
387fa574e6e25896bd3348474ec947dde70f3b11. Local changes target NPU2/AIE2P, use all
32 compute tiles, and replace an incompatible parallel codebook lookup with
explicit gathers. See LICENSE and THIRD_PARTY.md.
