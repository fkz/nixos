// Run once with GGML_HIP_DISABLE_Q6_PAIR=1 and once without it; compare
// the binary float outputs. Covers optimized shapes and fallback paths.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <fstream>
#include <random>
#include <utility>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s BACKEND_LIBRARY DEVICE OUTPUT_FILE\n", argv[0]);
        return 1;
    }
    if (!ggml_backend_load(argv[1])) { return 2; }
    const auto device = ggml_backend_dev_by_name(argv[2]);
    if (!device) { return 3; }
    auto backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) { return 4; }
    std::ofstream output(argv[3], std::ios::binary);
    if (!output) { return 5; }
    for (auto type : {GGML_TYPE_Q6_K, GGML_TYPE_Q5_K}) {
        for (int tokens : {1, 2}) {
            for (auto shape : {std::pair<int, int>{6144, 2560}, {2560, 10240},
                               {2560, 512}, {2560, 640}, {2560, 6144}, {2560, 12288},
                               {2560, 1}, {2560, 639}, {2560, 641}, {1280, 513}}) {
                const int cols = shape.first, rows = shape.second;
                auto ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
                auto weights = ggml_new_tensor_2d(ctx, type, cols, rows);
                auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, tokens);
                auto result = ggml_mul_mat(ctx, weights, input);
                auto graph = ggml_new_graph_custom(ctx, 32, false);
                ggml_build_forward_expand(graph, result);
                auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                if (!buffer) { return 6; }
                std::mt19937 rng(1234);
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                std::vector<float> raw(cols * rows), values(cols * tokens);
                for (auto & v : raw) { v = dist(rng); }
                for (auto & v : values) { v = dist(rng); }
                std::vector<char> quantized(ggml_nbytes(weights));
                ggml_quantize_chunk(type, raw.data(), quantized.data(), 0, rows, cols, nullptr);
                ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
                ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 7; }
                ggml_backend_synchronize(backend);
                std::vector<float> actual(rows * tokens);
                ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
                output.write(reinterpret_cast<const char *>(actual.data()), actual.size() * sizeof(float));
                std::printf("%s [%d,%d], tokens=%d\n", ggml_type_name(type), cols, rows, tokens);
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
            }
        }
    }
    ggml_backend_free(backend);
    return output ? 0 : 8;
}
