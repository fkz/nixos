# Source attribution

`q5_k/mm/q5_mm.cc` adapts the BF16 `matmul_vectorized_4x4` implementation from [`aie_kernels/aie2/mm.cc`](https://github.com/Xilinx/mlir-aie/blob/20a9c2fb2dca34d9d0f6f0c91793107d737423c3/aie_kernels/aie2/mm.cc). The original file is Copyright (C) 2025 Advanced Micro Devices, Inc., under Apache-2.0 WITH LLVM-exception. The full [license text](LICENSES/Apache-2.0-WITH-LLVM-exception.txt) is retained.

Local changes specialize operand layouts and pointer traversal, remove the transpose of each B vector from the multiplication loop, add an output-row offset, and make tracing and partial-storage type selectable. The upstream implementation already provides 4 x 4 register blocking, BF16/FP32 outputs, and event0/event1 markers.

`iq1_s/mm/iq1_s_lut.h` contains the IQ1_S codebook from llama.cpp's `ggml/src/ggml-common.h`, packed into signed four-bit entries. It is Copyright (c) 2023-2026 The ggml authors, under the [MIT license](LICENSE).

mlir-aie, Peano, the AIE API, XRT, and llama.cpp remain external dependencies. Their sources and binary distributions retain their respective licenses.
