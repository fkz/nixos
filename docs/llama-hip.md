# llama.cpp HIP hyper-connection projection optimization

The HIP build includes a narrow fast path for Flash-Next's BF16 HC up
projection: contiguous `[320, 10240]` weights, one input token, no expert IDs,
no fused gate or bias, one channel/sample, RDNA 3.5, and wave32.
It uses one wave per output row and two rows per block, replacing the generic
five-wave/shared-memory reduction with a wave-local float reduction.
Weights remain BF16 and accumulation remains float; reduction order changes,
so results are numerically close rather than bit-identical.
Other shapes, prefill batches, fused operations, and architectures keep their
existing paths. The Vulkan build does not include this patch.

## Measured results

Measured 2026-10-22 on Radeon 840M, ROCm 7.2.3 with the command-scoped gfx1150
compatibility override, llama.cpp v0.7.0 (`620d222`), GSQ-RCO IQ3_XXS Flash-Next.
Same patched binary, optimization enabled versus disabled, full GPU offload,
lazy mode on, F16 KV, flash attention auto, no draft model. `llama-bench`
used 32 synthetic decode tokens, three repetitions at each context depth.
These are throughput measurements, not downstream quality benchmarks.

| Context depth | Disabled tokens/s | Enabled tokens/s | Gain |
|---:|---:|---:|---:|
| 0 | 5.91 | 7.44 | 25.9% |
| 2048 | 5.49 | 6.88 | 25.3% |

The isolated installed-backend `[320,10240]` BF16 projection improved from
approximately 0.86 to 0.10 ms per synchronous evaluation. Absolute isolated
timings differ from in-model GPU timings; the table above is the full-model
result. Server sampling, real text, context length, and concurrent workloads
can change observed throughput.

## Use and A/B testing

The optimization is on by default in the `llama-hip` package:

```sh
nix shell /home/fabian/nixos#llama-hip -c llama-server-hip \
  -hf ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_XXS \
  -ngl 99 --lazy-mode on --reasoning-effort low
```

Set `GGML_HIP_DISABLE_HC_UP=1` before starting a HIP command to use the original
kernel. This switch is read once per process. Do not confuse it with
`GGML_CUDA_DISABLE_GRAPHS`, which disables graph execution instead.
For system-wide installation, commit/push first and run `sudo nixos-update`.

## Numerical validation

`tests/llama-hip-hc-up.cpp` checks random BF16 matrices and float vectors against
a double-precision dot-product reference. It covers the optimized shape,
nearby fallback dimensions, the reverse projection, and one/two input tokens.
Maximum absolute error on the fast-path case was about `2.8e-6`; the allowed
scaled error is `1e-4`. All fallback outputs matched the original kernel's
checked outputs. This does not assert bitwise equality or unchanged benchmark
quality for every possible model input.

Build and run the validation harness from this repository:

```sh
task_src=$(nix eval --raw .#llama-cpp-hip.src.outPath)
task_out=$(nix build .#llama-cpp-hip --no-link --print-out-paths)
nix develop .#llama-cpp-hip -c c++ -O2 \
  -I"$task_src/ggml/include" tests/llama-hip-hc-up.cpp \
  -L"$task_out/lib" -Wl,-rpath,"$task_out/lib" \
  -lggml -lggml-base -o /tmp/check-hip-hc-up
HSA_OVERRIDE_GFX_VERSION=11.5.0 /tmp/check-hip-hc-up \
  "$task_out/bin/libggml-hip.so" ROCm0
```
