#include <aie_api/aie.hpp>

using MMUL = aie::mmul<4, 8, 8, int8, int8, acc32>;

__attribute__((always_inline)) static inline void store_dot(aie::vector<int32, 32> dots, aie::vector<uint32, 32> bs, const int8 * a, bfloat16 * dst) {
    const auto as = aie::load_v<8>(reinterpret_cast<const float *>(a + 2048)).grow_replicate<32>().cast_to<uint32>();
    // abs(dot) < 2^22; combine the two power-of-two scales in the conversion bias.
    const auto bias = aie::add(aie::add(as, bs), uint32(0xcc400000)).cast_to<int32>();
    const auto scaled = aie::sub(aie::add(dots, bias).cast_to<float>(), bias.cast_to<float>());
    const auto old = aie::accum<accfloat, 32>(aie::transpose(aie::load_v<32>(dst), 8, 4)).to_vector<float>();
    const auto value = aie::accum<accfloat, 32>(aie::add(old, scaled)).to_vector<bfloat16>();
    aie::store_v(dst, aie::transpose(value, 4, 8));
}

extern "C" void iq4_xs_packed_zero(bfloat16 * c) {
    for (unsigned i = 0; i < DIM_OUT_M * DIM_N; i += 32) aie::store_v(c + i, aie::zeros<bfloat16, 32>());
}

static inline int subscale(const uint8_t * block, unsigned group) {
    uint16_t high;
    __builtin_memcpy(&high, block + 2, sizeof(high));
    return int(((block[4 + group / 2] >> (4 * (group % 2))) & 15) | (((high >> (2 * group)) & 3) << 4)) - 32;
}

__attribute__((always_inline)) static inline aie::vector<int8, 32> block_peaks(const uint8_t * q) {
    aie::vector<uint32, 8> packed;
#pragma clang loop unroll(full)
    for (unsigned row = 0; row < 4; ++row) {
        uint16_t high;
        uint32_t low;
        __builtin_memcpy(&high, q + row * 272 + 2, sizeof(high));
        __builtin_memcpy(&low, q + row * 272 + 4, sizeof(low));
#pragma clang loop unroll(full)
        for (unsigned parity = 0; parity < 2; ++parity) {
            // Expand the high two bits into four byte lanes.
            uint32_t h = (high >> (2 * parity)) & 0x3333;
            h = (h | (h << 8)) & 0x00ff00ff;
            h = (h | (h << 4)) & 0x0f0f0f0f;
            packed.set(((low >> (4 * parity)) & 0x0f0f0f0f) | (h << 4), 2 * row + parity);
        }
    }
    // Read each row's maximum from lanes 0, 8, 16 and 24.
    auto values = aie::abs(aie::sub(packed.cast_to<int8>(), int8(32)));
    values = aie::max(values, aie::shuffle_down_rotate(values, 4));
    values = aie::max(values, aie::shuffle_down_rotate(values, 2));
    return aie::max(values, aie::shuffle_down_rotate(values, 1));
}

extern "C" void iq4_xs_packed_decode(const uint8_t * __restrict q, int8 * __restrict b, float * __restrict scales, unsigned step, const int16_t * lut_ab, const int16_t * lut_cd) {
#if IQ4_TRACE
    event0();
#endif
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);
    using LUT = aie::lut<4, int16, int16>;
    const LUT lut(16, lut_ab, lut_cd);
    aie::parallel_lookup<uint16, LUT, aie::lut_oor_policy::truncate> lookup(lut);
    for (unsigned col = 0; col < DIM_N; col += 4) {
        const auto peaks = block_peaks(q + col * 272 + step * 136);
        int mantissa[4], shift[4];
#pragma clang loop unroll(full)
        for (unsigned row = 0; row < 4; ++row) {
            const uint8_t * block = q + (col + row) * 272 + step * 136;
            uint16_t hd;
            __builtin_memcpy(&hd, block, sizeof(hd));
            const int exponent = (hd & 0x7c00) ? int((hd >> 10) & 31) - 25 : -24;
            const int magnitude = (hd & 1023) | ((hd & 0x7c00) ? 1024 : 0);
            mantissa[row] = (hd & 0x8000) ? -magnitude : magnitude;
            const unsigned peak = unsigned(peaks.get(row * 8));
            // The two factors of 127 cancel in the INT8 range bound.
            const unsigned bound = unsigned(magnitude) * peak;
            const int s = bound <= 1 ? 0 : 32 - __builtin_clz(bound - 1);
            shift[row] = s;
            union { unsigned u; float f; } power;
            power.u = unsigned(exponent + s + 127) << 23;
            scales[col + row] = power.f;
        }
        for (unsigned group = 0; group < 8; ++group) {
            aie::vector<int8, 32> decoded[4];
#pragma clang loop unroll(full)
            for (unsigned row = 0; row < 4; ++row) {
                const uint8_t * block = q + (col + row) * 272 + step * 136;
                const auto packed = aie::load_unaligned_v<16>(block + 8 + group * 16).unpack<uint16>();
                const auto hi = aie::accum<acc32, 16>(aie::bit_and(uint16(240), packed)).to_vector<uint16>(4);
                const auto codes = aie::concat(aie::bit_and(uint16(15), packed), hi);
                const int factor = mantissa[row] * subscale(block, group);
                const auto grid = lookup.fetch(codes);
                const auto signed_grid = aie::select(grid, aie::neg(grid), aie::mask<32>::from_uint32(factor < 0 ? 0xffffffffu : 0));
                decoded[row] = aie::mul(signed_grid, uint16(factor < 0 ? -factor : factor)).to_vector<int8>(shift[row]);
            }
            const auto p01 = aie::interleave_zip(decoded[0], decoded[1], 8);
            const auto p23 = aie::interleave_zip(decoded[2], decoded[3], 8);
            const auto lo = aie::interleave_zip(p01.first, p23.first, 16);
            const auto hi = aie::interleave_zip(p01.second, p23.second, 16);
            const aie::vector<int8, 32> packed[4] = {lo.first, lo.second, hi.first, hi.second};
#pragma clang loop unroll(full)
            for (unsigned kk = 0; kk < 4; ++kk) aie::store_v(b + (group * 4 + kk) * DIM_N * 8 + col * 8, packed[kk]);
        }
    }
#if IQ4_TRACE
    event1();
#endif
}

extern "C" void iq4_xs_packed_mm(const int8 * __restrict a, const int8 * __restrict b, const float * __restrict scales, bfloat16 * __restrict c, unsigned row_offset) {
#if IQ4_TRACE
    event0();
#endif
    aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
    for (unsigned nr = 0; nr < DIM_N / 4; nr += 4) {
#pragma clang loop unroll(disable)
        for (unsigned mr = 0; mr < DIM_M / 8; mr += 2) {
            MMUL accum[4][2];
#pragma clang loop unroll(full)
            for (unsigned i = 0; i < 4; ++i) {
                const auto weights = aie::load_v<32>(b + (nr + i) * 32);
#pragma clang loop unroll(full)
                for (unsigned j = 0; j < 2; ++j) accum[i][j].mul(weights, aie::load_v<64>(a + (mr + j) * 2112));
            }
            const int8 * next_b = b + nr * 32 + DIM_N * 8;
            const int8 * next_a = a + mr * 2112 + 64;
            for (unsigned kk = 1; kk < 32; ++kk) {
                aie::vector<int8, 32> weights[4];
                aie::vector<int8, 64> activations[2];
#pragma clang loop unroll(full)
                for (unsigned i = 0; i < 4; ++i) weights[i] = aie::load_v<32>(next_b + i * 32);
#pragma clang loop unroll(full)
                for (unsigned j = 0; j < 2; ++j) activations[j] = aie::load_v<64>(next_a + j * 2112);
#pragma clang loop unroll(full)
                for (unsigned i = 0; i < 4; ++i) {
#pragma clang loop unroll(full)
                    for (unsigned j = 0; j < 2; ++j) accum[i][j].mac(weights[i], activations[j]);
                }
                next_b += DIM_N * 8;
                next_a += 64;
            }
#pragma clang loop unroll(full)
            for (unsigned i = 0; i < 4; ++i) {
                aie::vector<uint32, 32> bs;
#pragma clang loop unroll(full)
                for (unsigned r = 0; r < 4; ++r) bs.insert<8>(r, aie::broadcast<float, 8>(scales[(nr + i) * 4 + r]).cast_to<uint32>());
#pragma clang loop unroll(full)
                for (unsigned j = 0; j < 2; ++j) {
                    auto * dst = c + ((nr + i) * (DIM_OUT_M / 8) + row_offset / 8 + mr + j) * 32;
                    store_dot(accum[i][j].to_vector<int32>(), bs, a + (mr + j) * 2112, dst);
                }
            }
        }
    }
#if IQ4_TRACE
    event1();
#endif
}

extern "C" void iq4_xs_packed_header(const uint8_t * q, uint32_t * header) {
    header[0] = unsigned(q[0]) | (unsigned(q[1]) << 8);
    header[1] = unsigned(q[2]) | (unsigned(q[3]) << 8);
}
