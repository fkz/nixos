# llama.cpp HIP decode optimizations

The HIP build includes a narrow fast path for Flash-Next's BF16 HC up
projection: contiguous `[320, 10240]` weights, one input token, no expert IDs,
no fused gate or bias, one channel/sample, RDNA 3.5, and wave32.
It uses one wave per output row and two rows per block, replacing the generic
five-wave/shared-memory reduction with a wave-local float reduction.
Weights remain BF16 and accumulation remains float; reduction order changes,
so results are numerically close rather than bit-identical.
Additional BF16 fast paths cover contiguous `[10240,320]` HC down projections
and `[2560,512]` router projections under the same single-token, single-channel,
no-fusion/no-expert-ID RDNA 3.5 restrictions. These use two float accumulators
per lane and two independent rows per block. Other shapes, multi-token prefill
batches, fused operations, and architectures keep their existing paths.
The Vulkan build does not include these patches.

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

## Follow-up BF16 optimization

A matching control keeps the first HC up optimization enabled and toggles only
the added HC down and router paths. The benchmark model and settings are the
same as above. All GPU comparisons must run without a concurrent build or
another inference process using the APU.

The built package was checked with four short, cooled ABBA-ordered runs:

| Run | Added paths | Tokens/s | Start / peak temperature |
|---:|---|---:|---:|
| 1 | Disabled | 5.904 | 74 / 94 °C |
| 2 | Enabled | 6.308 | 75 / 94 °C |
| 3 | Enabled | 6.412 | 75 / 93 °C |
| 4 | Disabled | 6.030 | 75 / 93 °C |

Means: **5.967 tokens/s disabled, 6.360 enabled**, a **6.6%** improvement. Each run used 32 generated benchmark tokens, one repetition, empty starting context, and the same model and settings. The previous HC up optimization remained enabled in every run.

This comparison uses the actual built package, without temporary kernel modules. Numerical checks passed for all 14 shape/token-count cases.

An earlier exploratory on/off comparison at context depths 0 and 2048 suggested gains of 5.5–6.5%. Later sustained verification became highly variable as the GPU sensor reached 97–98°C and the active clock dropped below its maximum; enabled samples ranged from about 3.6 to 6.1 tokens/s. Those unstable long runs do **not** establish a sustained improvement or a reliable HIP-versus-Vulkan ranking. The cooled short-run comparison above is the qualified result. Absolute speeds across this section and the initial optimization section should not be compared directly.

Earlier Q6_K experiments with two waves per row and with independently packed rows
were also tested but did not show a useful isolated improvement. They are not
included in the package. The added BF16 kernels use the original BF16 weights
and float accumulation; they do not change the model quantization.

## Use and A/B testing

The optimization is on by default in the `llama-hip` package:

```sh
nix shell /home/fabian/nixos#llama-hip -c llama-server-hip \
  -hf ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_XXS \
  -ngl 99 --lazy-mode on --reasoning-effort low
```

Set `GGML_HIP_DISABLE_HC_UP=1` before starting a HIP command to use the original
HC up kernel. Set `GGML_HIP_DISABLE_DENSE_BF16=1` to disable the added HC down
and router paths. Set both to disable the custom BF16 projection kernels.
These switches are read once per process. Do not confuse it with
`GGML_CUDA_DISABLE_GRAPHS`, which disables graph execution instead.
For system-wide installation, commit/push first and run `sudo nixos-update`.

## Numerical validation

`tests/llama-hip-hc-up.cpp` checks random BF16 matrices and float vectors against
a double-precision dot-product reference. It covers the optimized shape,
nearby fallback dimensions, the reverse projection, and one/two input tokens.
The initial HC up fast path had maximum absolute error about `2.8e-6`.
The expanded checks include the router shape and nearby row/column dimensions;
maximum absolute error on the longer HC down dot products was about `2.8e-5`,
with scaled error about `1.1e-5`, below the `1e-4` bound. These errors are
against a double-precision dot product of the BF16 weights and float inputs.
Fallback checks also passed. This does not assert bitwise equality or unchanged benchmark
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

## Q6_K follow-up (2026-10-23)

The Q6_K fast path keeps the original Q6_K weights, Q8_1 activation quantization,
and float accumulation. Each lane evaluates two Q6_K contributions with
independent accumulators; one wave owns an output row. Blocks contain two
independent rows, or eight for small output matrices. It avoids the generic
multi-shape kernel's indexing and fusion machinery for this narrow case.

The guard requires RDNA 3.5 / wave32, a single token/channel/sample, no expert
IDs, fusion or shared-expert operation, contiguous weight rows, and input
width 2560 or 6144. Other widths, multi-token batches and other weight types
use the original path. There is no model/state memory-layout change.
`GGML_HIP_DISABLE_Q6_PAIR=1` disables only this optimization, once per process.

A temporary backend overlay was compared with the same overlay disabled, in
cooled ABBA order. The existing BF16 optimizations stayed enabled. Settings:
64 decode tokens, two repetitions at each depth, full GPU offload, lazy mode
on, F16 KV, flash attention auto, six CPU threads, no draft model. Means of
four throughput samples per condition:

| Context depth | Q6 path disabled tokens/s | Enabled tokens/s | Gain |
|---:|---:|---:|---:|
| 0 | 7.372 | 7.584 | 2.9% |
| 2048 | 6.931 | 7.199 | 3.9% |

Run means (depth 0 / depth 2048): off 7.327 / 6.984, on 7.622 / 7.188,
on 7.546 / 7.209, off 7.418 / 6.877. Starting GPU sensor readings were
64–66°C; peaks were 86–97°C. The last depth-2048 control samples still had
noticeable spread (6.672–7.083). These are short benchmark results, not a
promise of the same sustained or server throughput. Earlier full-model runs
with a competing GPU workload were discarded after the user paused it.

Isolated synchronous backend tests suggested roughly 14–18% lower time on
the larger Q6_K products. Those exploratory tests preceded pausing the other
GPU workload and are less controlled than the full-model comparison above.
The full-model gain is smaller because Q6_K is only one part of decode.
The earlier profile measured about 18 ms/token for HIP Q6_K, versus 12.6 ms
for Vulkan, with differing profiler overhead. The 71 Q6_K products represent
about 864 MB of weight bytes/token; bytes divided by kernel time includes
unpacking and arithmetic, and is not a direct RAM-bandwidth measurement.

### Q6 numerical regression check

`tests/llama-hip-q6.cpp` writes all outputs for 40 cases: all six model shapes,
odd/small output row counts, an unsupported input width, one/two tokens,
and Q5_K controls. Compare separate processes with the Q6 optimization
on/off using `tests/check-llama-hip-q6.py`. The test requires finite outputs,
scaled error below `1e-4` on optimized cases, and exact equality on fallback
cases. It compares against the previous HIP path; it is not a downstream
model-quality benchmark. The exploratory module passed all 205,068 outputs,
with maximum absolute difference `1.53e-5` and scaled difference `8.67e-6`.
The final Nix-built package passed the same 40-case comparison without any
temporary preload module, with the same maximum differences. The complete
model also loaded and generated 32 tokens successfully (6.949 tokens/s in a
single post-build functional run). That unpaired run was not cooled or used
to estimate the performance gain above. Both the package build and full
NixOS toplevel derivation evaluation succeeded; no system activation was done.

```sh
task_src=$(nix eval --raw .#llama-cpp-hip.src.outPath)
task_out=$(nix build .#llama-cpp-hip --no-link --print-out-paths)
nix develop .#llama-cpp-hip -c c++ -O2 \
  -I"$task_src/ggml/include" tests/llama-hip-q6.cpp \
  -L"$task_out/lib" -Wl,-rpath,"$task_out/lib" \
  -lggml -lggml-base -o /tmp/check-hip-q6
GGML_HIP_DISABLE_Q6_PAIR=1 HSA_OVERRIDE_GFX_VERSION=11.5.0 \
  /tmp/check-hip-q6 "$task_out/bin/libggml-hip.so" ROCm0 /tmp/q6-off.bin
HSA_OVERRIDE_GFX_VERSION=11.5.0 \
  /tmp/check-hip-q6 "$task_out/bin/libggml-hip.so" ROCm0 /tmp/q6-on.bin
python3 tests/check-llama-hip-q6.py /tmp/q6-off.bin /tmp/q6-on.bin
```
