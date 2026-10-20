"""Independent IQ3_XXS and INT8/BF16 reference; adapted from phhusson/amd-npu-gguf."""
import ctypes, os, re, shutil
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16 as BF16
HERE=Path(__file__).resolve().parent
def base_library():
    if os.environ.get("LLAMA_BASE_LIB"): return os.environ["LLAMA_BASE_LIB"]
    return str(Path(shutil.which("llama")).resolve().parents[1]/"lib/libggml-base.so")
GRID=np.array([int(x,16) for x in re.findall(r'0x[0-9a-fA-F]+',(HERE/'iq3_xxs_lut.h').read_text())],dtype='<u4').view(np.uint8).reshape(256,4)
SIGNS=np.array([i|((i.bit_count()&1)<<7) for i in range(128)],dtype=np.uint8)
def pack_activations(x):
    M, K = x.shape
    blocks = x.reshape(M, K // 256, 256)
    peak = np.maximum(np.max(np.abs(blocks), axis=-1), 2.0 ** -60)
    scales = np.exp2(np.ceil(np.log2(peak.astype(np.float64) / 127))).astype(np.float32)
    codes = np.clip(np.rint(blocks / scales[..., None]), -127, 127).astype(np.int8)
    packed = np.zeros((K // 256, M // 8, 2112), dtype=np.int8)
    packed[..., :2048] = codes.reshape(M // 8, 8, K // 256, 32, 8).transpose(2, 0, 3, 4, 1).reshape(K // 256, M // 8, 2048)
    packed[..., 2048:2080] = scales.T.copy().reshape(K // 256, M // 8, 8).view(np.int8)
    return packed.reshape(-1), codes.astype(np.float32), scales


def matmul(xcodes, xscales, wcodes, wscales):
    xe, we = np.log2(xscales), np.log2(wscales)
    assert np.all(xe.min(axis=0) + we.min(axis=0) >= -126)
    assert np.all(xe.max(axis=0) + we.max(axis=0) <= 104)
    out = np.zeros((xcodes.shape[0], wcodes.shape[0]), dtype=np.float32)
    for k in range(xcodes.shape[1]):
        dots = np.ascontiguousarray(xcodes[:, k]) @ np.ascontiguousarray(wcodes[:, k]).T
        out = (out + (dots * wscales[:, k]) * xscales[:, k, None]).astype(BF16).astype(np.float32)
    return out

def decode(q):
    h = q[..., :2].copy().view('<f2').astype(np.float64) * .25
    assert np.isfinite(h).all()
    words = q[..., 66:98].copy().view('<u4')
    sc = (1 + 2 * (words >> 28)).astype(np.int32)
    coeff = np.empty((*q.shape[:2], 256), dtype=np.int32)
    for group in range(8):
        values = GRID[q[..., 2 + group * 8:10 + group * 8]].reshape(*q.shape[:2], 32).astype(np.int32)
        selectors = (words[..., group, None] >> (7 * np.arange(4))) & 127
        signs = ((SIGNS[selectors][..., None] >> np.arange(8)) & 1).reshape(*q.shape[:2], 32)
        coeff[..., group * 32:(group + 1) * 32] = values * (1 - 2 * signs) * sc[..., group, None]
    native = (h * coeff).astype(np.float32)
    lib = ctypes.CDLL(base_library())
    fn = lib.dequantize_row_iq3_xxs
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None
    ggml = np.empty_like(native)
    fn(q.ctypes.data, ggml.ctypes.data, native.size)
    np.testing.assert_array_equal(native, ggml)
    peak = np.abs(h[..., 0]) * np.max(sc, axis=-1) * 62
    hd = q[..., :2].copy().view('<u2')[..., 0].astype(np.int32)
    smallest = np.maximum((hd >> 10) & 31, 1) - 27
    exponent = np.maximum(smallest, np.ceil(np.log2(np.maximum(peak / 127, 2.0 ** -100))))
    scales = np.exp2(exponent).astype(np.float32)
    codes = np.clip(np.rint((h * coeff) / scales[..., None]), -127, 127).astype(np.float32)
    return native.reshape(q.shape[0], -1), codes, scales
