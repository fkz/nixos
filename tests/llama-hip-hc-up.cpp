#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <utility>
// Usage: check-matvec /path/to/libggml-hip.so ROCm0
int main(int argc, char ** argv) {
    if (argc != 3 || !ggml_backend_load(argv[1])) return 1;
    auto dev = ggml_backend_dev_by_name(argv[2]);
    if (!dev) return 2;
    auto backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return 3;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    // Fast path, nearby fallback shapes, and multi-token fallback.
    for (auto shape : {std::pair<int,int>{320,10240}, {320,10238}, {322,10240}, {10240,320}}) {
        for (int tokens : {1,2}) {
            int cols = shape.first, rows = shape.second;
            auto ctx = ggml_init({16*1024*1024, nullptr, true});
            auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, cols, rows);
            auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, tokens);
            auto y = ggml_mul_mat(ctx, w, x);
            auto graph = ggml_new_graph_custom(ctx, 32, false);
            ggml_build_forward_expand(graph, y);
            auto buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) return 4;
            std::vector<ggml_bf16_t> weights(size_t(cols)*rows);
            std::vector<float> input(cols*tokens), output(rows*tokens);
            for (auto & v : weights) v = ggml_fp32_to_bf16(dist(rng));
            for (auto & v : input) v = dist(rng);
            ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 5;
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(y, output.data(), 0, ggml_nbytes(y));
            double max_abs = 0, max_scaled = 0;
            for (int t = 0; t < tokens; ++t) {
                for (int r = 0; r < rows; ++r) {
                    double ref = 0;
                    for (int c = 0; c < cols; ++c) {
                        ref += double(ggml_bf16_to_fp32(weights[size_t(r)*cols+c])) * input[t*cols+c];
                    }
                    double err = std::abs(output[t*rows+r] - ref);
                    if (!std::isfinite(output[t*rows+r])) return 6;
                    max_abs = std::fmax(max_abs, err);
                    max_scaled = std::fmax(max_scaled, err/(1+std::abs(ref)));
                }
            }
            std::printf("BF16 [%d,%d] tokens=%d max_abs=%.9g max_scaled=%.9g\n", cols, rows, tokens, max_abs, max_scaled);
            if (max_scaled > 1e-4) return 7;
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
}
