// Optional XDNA2 prefill bridge. Native IQ3_XXS weights; INT8/BF16 approximate math.
#include "ggml-alloc.h"
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_hw_context.h>
#include <fstream>
#include <mutex>
#include <cmath>
#include <memory>
#include <map>

namespace {
constexpr int npu_k = 2560, npu_n = 640;
static int npu_env_int(const char * name, int fallback) {
    const char * s = std::getenv(name);
    if (!s || !*s) return fallback;
    char * end = nullptr;
    long value = std::strtol(s, &end, 10);
    return end && !*end && value >= 1 && value <= 65536 ? int(value) : fallback;
}
static bool npu_enabled() {
    const char * value = std::getenv("GGML_VK_NPU_PREFILL");
    return value && std::strcmp(value, "1") == 0;
}
static bool npu_candidate(const ggml_tensor * node) {
    if (!npu_enabled() || node->op != GGML_OP_MUL_MAT_ID) return false;
    const auto * w = node->src[0]; const auto * a = node->src[1]; const auto * ids = node->src[2];
    return w && a && ids && w->type == GGML_TYPE_IQ3_XXS && a->type == GGML_TYPE_F32 &&
        ids->type == GGML_TYPE_I32 && node->type == GGML_TYPE_F32 &&
        w->ne[0] == npu_k && w->ne[1] == npu_n && w->ne[3] == 1 &&
        a->ne[0] == npu_k && a->ne[2] == ids->ne[1] && a->ne[3] == 1 &&
        ids->ne[1] > 1 && ids->ne[2] == 1 && ids->ne[3] == 1 &&
        a->ne[1] > 0 && ids->ne[0] % a->ne[1] == 0 &&
        ggml_is_contiguous(w) && ggml_is_contiguous(a) && ids->nb[0] == sizeof(int32_t) && ggml_is_contiguous(node);
}
static void npu_pack(const float * x, int8_t * out, int M) {
    constexpr int K = npu_k;
    const int threads = npu_env_int("GGML_VK_NPU_PACK_THREADS", 6);
    (void) threads;
#pragma omp parallel for collapse(2) num_threads(threads)
    for (int kb = 0; kb < K/256; ++kb) for (int mb = 0; mb < M/8; ++mb) {
        alignas(64) int8_t codes[8][256]; float scales[8];
        for (int r = 0; r < 8; ++r) {
            const float * a = x + (mb*8+r)*K + kb*256;
            float peak = 0x1p-60f;
#pragma omp simd reduction(max:peak)
            for (int k = 0; k < 256; ++k) peak = std::max(peak, std::fabs(a[k]));
            int e = std::ilogb(peak)-6; float s = std::ldexp(1.0f, e);
            if (127.0f*s < peak) s *= 2;
            scales[r] = s; const float inv = 1.0f/s;
#pragma omp simd
            for (int k = 0; k < 256; ++k)
                codes[r][k] = int8_t(std::max(-127.0f, std::min(127.0f, std::nearbyint(a[k]*inv))));
        }
        int8_t * dst = out + (kb*(M/8)+mb)*2112;
        for (int k = 0; k < 256; ++k) for (int r = 0; r < 8; ++r)
            dst[(k/8)*64+(k%8)*8+r] = codes[r][k];
        std::memcpy(dst+2048, scales, sizeof(scales));
    }
}
static float npu_bf16(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16; float result;
    std::memcpy(&result, &bits, 4); return result;
}
struct npu_plan {
    int M;
    xrt::xclbin binary;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo instructions, a, b, c;
    uint32_t instruction_bytes;
    npu_plan(xrt::device & device, const std::string & directory, int rows) : M(rows),
        binary(directory + "/m" + std::to_string(rows) + ".xclbin"),
        context(device, device.register_xclbin(binary)), kernel(context, "MLIR_AIE") {
        std::ifstream stream(directory + "/m" + std::to_string(rows) + ".bin", std::ios::binary);
        if (!stream) throw std::runtime_error("cannot open NPU instruction file");
        std::vector<char> data((std::istreambuf_iterator<char>(stream)), {});
        if (data.empty() || data.size() > UINT32_MAX) throw std::runtime_error("invalid NPU instruction size");
        instruction_bytes = uint32_t(data.size());
        instructions = xrt::bo(device, data.size(), xrt::bo::flags::cacheable, kernel.group_id(1));
        std::memcpy(instructions.map<void *>(), data.data(), data.size());
        instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        a = xrt::bo(device, size_t(M)*npu_k*33/32, xrt::bo::flags::host_only, kernel.group_id(3));
        b = xrt::bo(device, size_t(npu_n)*(npu_k/256)*98, xrt::bo::flags::host_only, kernel.group_id(4));
        c = xrt::bo(device, size_t(M)*npu_n*2, xrt::bo::flags::host_only, kernel.group_id(5));
    }
    void run(const float * input, const uint8_t * weights, std::vector<float> & output) {
        std::memset(a.map<void *>(), 0, a.size());
        npu_pack(input, a.map<int8_t *>(), M);
        std::memcpy(b.map<void *>(), weights, b.size());
        a.sync(XCL_BO_SYNC_BO_TO_DEVICE); b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = kernel(3, instructions, instruction_bytes, a, b, c);
        if (run.wait(10000) != ERT_CMD_STATE_COMPLETED) {
            run.abort(); throw std::runtime_error("NPU execution did not complete");
        }
        c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const auto * values = c.map<const uint16_t *>();
        output.resize(size_t(M)*npu_n);
        for (size_t i = 0; i < output.size(); ++i) output[i] = npu_bf16(values[i]);
    }
};
struct npu_runtime {
    std::mutex mutex;
    std::unique_ptr<xrt::device> device;
    std::unique_ptr<npu_plan> small, large;
    bool failed = false;
    npu_plan & plan(int M) {
        const char * directory = std::getenv("GGML_VK_NPU_KERNEL_DIR");
        if (!directory || !*directory) throw std::runtime_error("GGML_VK_NPU_KERNEL_DIR is not set");
        if (!device) device = std::make_unique<xrt::device>(0);
        auto & slot = M == 64 ? small : large;
        if (!slot) slot = std::make_unique<npu_plan>(*device, directory, M);
        return *slot;
    }
};
struct npu_route { int slot, token, expert; };
struct npu_gpu_work {
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~npu_gpu_work() { if (buffer) ggml_backend_buffer_free(buffer); if (context) ggml_free(context); }
};
// Called only between completed Vulkan subgraphs. A failure leaves the node
// untouched and the caller executes it normally on Vulkan.
static bool npu_compute(ggml_backend_t backend, ggml_tensor * node) {
    static npu_runtime runtime;
    std::lock_guard<std::mutex> guard(runtime.mutex);
    if (runtime.failed) return false;
    const auto * w = node->src[0]; const auto * a = node->src[1]; const auto * ids = node->src[2];
    try {
        std::vector<int32_t> id_data(ggml_nelements(ids));
        std::vector<uint8_t> id_bytes(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, id_bytes.data(), 0, id_bytes.size());
        for (int64_t t = 0; t < ids->ne[1]; ++t) for (int64_t s = 0; s < ids->ne[0]; ++s)
            std::memcpy(&id_data[size_t(t)*ids->ne[0]+s],
                        id_bytes.data()+size_t(t)*ids->nb[1]+size_t(s)*ids->nb[0], sizeof(int32_t));
        const int slots = int(ids->ne[0]), tokens = int(ids->ne[1]);
        const int minimum = npu_env_int("GGML_VK_NPU_MIN_EXPERT_TOKENS", 256);
        std::map<int, std::vector<npu_route>> groups;
        for (int t = 0; t < tokens; ++t) for (int s = 0; s < slots; ++s) {
            int expert = id_data[size_t(t)*slots+s];
            if (expert < 0 || expert >= w->ne[2]) return false;
            groups[expert].push_back({s,t,expert});
        }
        size_t selected = 0;
        for (const auto & g : groups) if (g.second.size() >= size_t(minimum)) selected += g.second.size();
        if (!selected) return false;
        // Initialize before allocating or modifying any graph output.
        runtime.plan(64); runtime.plan(512);
        std::vector<float> input(ggml_nelements(a));
        ggml_backend_tensor_get(a, input.data(), 0, ggml_nbytes(a));
        for (float v : input) if (!std::isfinite(v) || std::fabs(v) > 1e20f) return false;
        std::vector<float> output(ggml_nelements(node));
        std::vector<npu_route> remaining;
        auto activation = [&](const npu_route & r) {
            return input.data() + (size_t(r.token)*a->ne[1]+r.slot%a->ne[1])*npu_k;
        };
        auto save = [&](const npu_route & r, const float * values) {
            std::memcpy(output.data()+(size_t(r.token)*slots+r.slot)*npu_n, values, npu_n*sizeof(float));
        };
        const size_t weight_bytes = size_t(npu_n)*(npu_k/256)*98;
        std::vector<uint8_t> weights(weight_bytes);
        std::vector<float> gathered, computed;
        size_t dispatches = 0;
        for (const auto & g : groups) {
            if (g.second.size() < size_t(minimum)) { remaining.insert(remaining.end(), g.second.begin(), g.second.end()); continue; }
            ggml_backend_tensor_get(w, weights.data(), size_t(g.first)*w->nb[2], weight_bytes);
            for (size_t off = 0; off < weight_bytes; off += 98) {
                uint16_t scale; std::memcpy(&scale, weights.data()+off, 2);
                if ((scale & 0x7c00) == 0x7c00) return false;
            }
            for (size_t off = 0; off < g.second.size();) {
                size_t count = std::min<size_t>(512, g.second.size()-off);
                int M = count <= 64 ? 64 : 512;
                gathered.assign(size_t(M)*npu_k, 0.0f);
                for (size_t r = 0; r < count; ++r) std::memcpy(gathered.data()+r*npu_k, activation(g.second[off+r]), npu_k*sizeof(float));
                runtime.plan(M).run(gathered.data(), weights.data(), computed);
                for (size_t r = 0; r < count; ++r) save(g.second[off+r], computed.data()+r*npu_n);
                off += count; ++dispatches;
            }
        }
        if (!remaining.empty()) {
            // Rebatch the remaining token/expert pairs into a one-slot GPU
            // MUL_MAT_ID. This avoids recomputing the NPU's expert rows.
            npu_gpu_work gpu;
            gpu.context = ggml_init({ 1024*1024, nullptr, true });
            if (!gpu.context) throw std::runtime_error("GPU remainder context allocation failed");
            auto * ga = ggml_new_tensor_3d(gpu.context, GGML_TYPE_F32, npu_k, 1, remaining.size());
            auto * gi = ggml_new_tensor_2d(gpu.context, GGML_TYPE_I32, 1, remaining.size());
            auto * go = ggml_mul_mat_id(gpu.context, const_cast<ggml_tensor *>(w), ga, gi);
            auto * graph = ggml_new_graph_custom(gpu.context, 32, false);
            ggml_build_forward_expand(graph, go);
            gpu.buffer = ggml_backend_alloc_ctx_tensors(gpu.context, backend);
            if (!gpu.buffer) throw std::runtime_error("GPU remainder buffer allocation failed");
            gathered.resize(remaining.size()*npu_k);
            std::vector<int32_t> gpu_ids(remaining.size());
            for (size_t r = 0; r < remaining.size(); ++r) {
                std::memcpy(gathered.data()+r*npu_k, activation(remaining[r]), npu_k*sizeof(float)); gpu_ids[r] = remaining[r].expert;
            }
            ggml_backend_tensor_set(ga, gathered.data(), 0, gathered.size()*sizeof(float));
            ggml_backend_tensor_set(gi, gpu_ids.data(), 0, gpu_ids.size()*sizeof(int32_t));
            if (ggml_backend_vk_graph_compute_impl(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("GPU remainder execution failed");
            ggml_backend_vk_synchronize(backend);
            computed.resize(remaining.size()*npu_n);
            ggml_backend_tensor_get(go, computed.data(), 0, computed.size()*sizeof(float));
            for (size_t r = 0; r < remaining.size(); ++r) save(remaining[r], computed.data()+r*npu_n);
            static_cast<ggml_backend_vk_context *>(backend->context)->device->diag_cgraph = nullptr;
        }
        ggml_backend_tensor_set(node, output.data(), 0, output.size()*sizeof(float));
        fprintf(stderr, "ggml_npu_prefill: %s: %zu/%zu routed rows on NPU, %zu dispatches; %zu rows on Vulkan\n", node->name, selected, id_data.size(), dispatches, remaining.size());
        return true;
    } catch (const std::exception & e) {
        runtime.failed = true;
        fprintf(stderr, "ggml_npu_prefill: disabled after error: %s; falling back to Vulkan\n", e.what());
        return false;
    }
}
} // namespace

static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    if (!npu_enabled()) return ggml_backend_vk_graph_compute_impl(backend, graph);
    int start = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (!npu_candidate(graph->nodes[i])) continue;
        if (i > start) {
            ggml_cgraph part = *graph; part.nodes = graph->nodes+start; part.n_nodes = i-start;
            auto status = ggml_backend_vk_graph_compute_impl(backend, &part);
            if (status != GGML_STATUS_SUCCESS) return status;
        }
        ggml_backend_vk_synchronize(backend);
        start = i;
        if (npu_compute(backend, graph->nodes[i])) start = i+1;
    }
    if (start < graph->n_nodes) {
        ggml_cgraph part = *graph; part.nodes = graph->nodes+start; part.n_nodes = graph->n_nodes-start;
        return ggml_backend_vk_graph_compute_impl(backend, &part);
    }
    return GGML_STATUS_SUCCESS;
}
