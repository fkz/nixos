#include <aie_api/aie.hpp>

static inline uint32_t scale_signs(const uint8_t * block, unsigned group) {
    uint32_t value;
    __builtin_memcpy(&value, block + 66 + group * 4, sizeof(value));
    return value;
}

extern "C" void iq3_xxs_packed_decode(const uint8_t * __restrict q, int8 * __restrict b, float * __restrict scales, unsigned step, const uint32_t * lut_ab, const uint32_t * lut_cd) {
#if IQ3_TRACE
    event0();
#endif
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);
    using LUT = aie::lut<4, uint32, uint32>;
    const LUT lut(256, lut_ab, lut_cd);
    aie::parallel_lookup<uint16, LUT, aie::lut_oor_policy::truncate> lookup(lut);
    for (unsigned col = 0; col < DIM_N; col += 4) {
        int mantissa[4], shift[4];
#pragma clang loop unroll(full)
        for (unsigned row = 0; row < 4; ++row) {
            const uint8_t * block = q + (col + row) * 196 + step * 98;
            uint16_t hd;
            __builtin_memcpy(&hd, block, sizeof(hd));
            const int exponent = (hd & 0x7c00) ? int((hd >> 10) & 31) - 27 : -26;
            const int magnitude = (hd & 1023) | ((hd & 0x7c00) ? 1024 : 0);
            mantissa[row] = (hd & 0x8000) ? -magnitude : magnitude;
            // The unsigned maximum word has the largest scale nibble.
            const unsigned peak = 1 + 2 * (aie::reduce_max(aie::load_unaligned_v<8>(reinterpret_cast<const uint32_t *>(block + 66))) >> 28);
            const unsigned bound = unsigned(magnitude) * peak * 62;
            int s = 25 - __builtin_clz(bound | 1);
            if (s < 0) s = 0;
            if ((127u << s) < bound) ++s;
            shift[row] = s;
            union { unsigned u; float f; } power;
            power.u = unsigned(exponent + s + 127) << 23;
            scales[col + row] = power.f;
        }
        for (unsigned group = 0; group < 8; ++group) {
            aie::vector<int8, 32> decoded[4];
#pragma clang loop unroll(full)
            for (unsigned row = 0; row < 4; ++row) {
                const uint8_t * block = q + (col + row) * 196 + step * 98;
                const auto indices = aie::load_unaligned_v<16>(block + 2 + group * 8).unpack<uint16>().extract<8>(0);
                aie::vector<uint32, 8> grid_words;
#pragma clang loop unroll(full)
                for (unsigned lane = 0; lane < 8; ++lane) {
                    const unsigned index = indices.get(lane);
                    grid_words.set(lut_ab[(index / 4) * 8 + index % 4], lane);
                }
                const auto grid = grid_words.cast_to<int8>();
                const uint32_t word = scale_signs(block, group);
                uint32_t signs = (word & 0x7f) | ((word & 0x3f80) << 1) | ((word & 0x1fc000) << 2) | ((word & 0x0fe00000) << 3);
                uint32_t parity = signs ^ (signs >> 4);
                parity ^= parity >> 2;
                parity ^= parity >> 1;
                signs |= (parity & 0x01010101) << 7;
                const auto signed_grid = aie::select(grid, aie::neg(grid), aie::mask<32>::from_uint32(signs));
                const auto product = aie::mul(signed_grid.unpack<int16>(), int16(1 + 2 * (word >> 28))).to_vector<int16>();
                decoded[row] = aie::mul(product, int16(mantissa[row])).to_vector<int8>(shift[row]);
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
#if IQ3_TRACE
    event1();
#endif
}
