"""Native IQ3_XXS matmul with packed INT8 activations and K256 scales."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16 as BF16
from aie.iron import Buffer, CompileTime, ExternalFunction, In, Kernel, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.device import Tile
from aie.iron.controlflow import range_
from aie.iron.dataflow.objectfifo import ObjectFifoLink
HERE = Path(__file__).resolve().parent
SHARED_MM = HERE / "kernel.cc"


@iron.jit(source_files=[HERE / 'iq3_xxs.cc', HERE / 'iq3_xxs_lut.h', SHARED_MM])
def matmul(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], m: CompileTime[int], am: CompileTime[int], n: CompileTime[int], cols: CompileTime[int], rows: CompileTime[int], cache_a: CompileTime[bool], a_depth: CompileTime[int], stack_size: CompileTime[int], l2_depth: CompileTime[int], trace_size: CompileTime[int], trace_worker: CompileTime[int]):
    assert M % (m * rows) == 0 and N % (n * cols) == 0 and K % 512 == 0
    assert m % am == 0 and am % 16 == 0 and n % 16 == 0 and cols * rows <= 32
    group_cols = max(1, cols // 8)
    host_cols = cols // group_cols
    lt = np.ndarray[(512,), np.dtype[np.uint32]]
    values = np.array([int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]+', (HERE / 'iq3_xxs_lut.h').read_text())], dtype=np.uint32)
    assert values.size == 256
    table = np.repeat(values.reshape(-1, 1, 4), 2, axis=1).reshape(-1)
    at = np.ndarray[(am // 8 * 2112,), np.dtype[np.int8]]
    bt = np.ndarray[(n * 196,), np.dtype[np.uint8]]
    dt = np.ndarray[(n * 256,), np.dtype[np.int8]]
    st = np.ndarray[(n,), np.dtype[np.float32]]
    ct = np.ndarray[(m * n,), np.dtype[BF16]]
    flags = [f'-DDIM_M={am}', f'-DDIM_OUT_M={m}', f'-DDIM_N={n}', f'-DIQ3_TRACE={int(trace_size > 0)}', f'-DIQ4_TRACE={int(trace_size > 0)}', '-ffp-contract=off']
    decode = ExternalFunction('iq3_xxs_packed_decode', source_file=str(HERE / 'iq3_xxs.cc'), arg_types=[bt, dt, st, np.int32, lt, lt], compile_flags=flags)
    mm = ExternalFunction('iq4_xs_packed_mm', source_file=str(SHARED_MM), arg_types=[at, dt, st, ct, np.int32], compile_flags=flags)
    zero = Kernel('iq4_xs_packed_zero', mm.object_file_name, [ct])
    afs, bfs, cfs, bhosts, chosts, workers = [], [], [], [], [], []
    for row in range(rows):
        al2 = np.ndarray[(m // 8 * 2112 * (K // 256 if cache_a else 1),), np.dtype[np.int8]]
        af = ObjectFifo(al2, name=f'A{row}', depth=1 if cache_a else l2_depth)
        if cache_a:
            aa = ObjectFifo(al2, name=f'a{row}', depth=1, consumer_obj_type=at, repeat_count=N // (n * cols))
        else:
            aa = ObjectFifo(al2, name=f'a{row}', depth=l2_depth, consumer_obj_type=at)
        ObjectFifoLink(af.cons(), aa.prod())
        afs.append((af, aa))
    for col in range(host_cols):
        bl2 = np.ndarray[(group_cols * n * 196,), np.dtype[np.uint8]]
        bf = ObjectFifo(bl2, name=f'B{col}', depth=1)
        bfs.extend(bf.cons().split(offsets=[i * n * 196 for i in range(group_cols)], tile=Tile(col, 1), obj_types=[bt] * group_cols, depths=[1] * group_cols))
        bhosts.append(bf)
        cl2 = np.ndarray[(group_cols * rows * m * n,), np.dtype[BF16]]
        cf = ObjectFifo(cl2, name=f'C{col}', depth=1, dims_to_stream=[(m // 8, 32), (8, 4), (n // 4, m * 4), (4, 1)])
        cc = cf.prod().join(offsets=[i * m * n for i in range(group_cols * rows)], tile=Tile(col, 1), obj_types=[ct] * (group_cols * rows), depths=[1] * (group_cols * rows))
        cfs.extend([cc[i * rows:(i + 1) * rows] for i in range(group_cols)])
        chosts.append(cf)

    def core(a, b, c, scratch, scales, lut_ab, lut_cd, decode, mm, zero):
        for _ in range_(M // (m * rows) * (N // (n * cols))):
            out = c.acquire(1)
            zero(out)
            for _ in range_(K // 512):
                raw = b.acquire(1)
                for step in range(2):
                    decode(raw, scratch, scales, step, lut_ab, lut_cd)
                    for ar in range(m // am):
                        av = a.acquire(1)
                        mm(av, scratch, scales, out, ar * am)
                        a.release(1)
                b.release(1)
            c.release(1)

    for row in range(rows):
        for col in range(cols):
            scratch = Buffer(dt, name=f'decode_{row}_{col}')
            scales = Buffer(st, name=f'scales_{row}_{col}')
            tile = Tile(col // group_cols, 2 + (col % group_cols) * rows + row)
            luts = [Buffer(lt, initial_value=table, name=f'lut_{row}_{col}_{bank}', tile=tile, address=4096 + bank * 16384) for bank in (0, 1)]
            worker = Worker(core, [afs[row][1].cons(depth=a_depth), bfs[col].cons(), cfs[col][row].prod(), scratch, scales, *luts, decode, mm, zero], tile=tile, stack_size=stack_size)
            workers.append(worker)

    def sequence(a, b, c, ah, bh, ch):
        drain_rows = rows * m if rows * m < 1024 else 256
        assert rows * m % drain_rows == 0
        drain_chunks = rows * m // drain_rows
        for rm in range(M // (m * rows)):
            pending = []
            for cn in range(N // (n * cols)):
                tg = TaskGroup()
                for col in range(host_cols):
                    first_col = cn * cols + col * group_cols
                    ch[col].drain(c, offset=rm * rows * m * N + first_col * n, sizes=[group_cols, drain_chunks, drain_rows, n], strides=[n, drain_rows * N if drain_chunks > 1 else 0, N, 1], wait=True, group=tg)
                    bh[col].fill(b, offset=first_col * n * (K // 512) * 196, sizes=[1, K // 512, group_cols * n, 196], strides=[0, 196, (K // 512) * 196, 1], group=tg)
                if not cache_a or cn == 0:
                    for row in range(rows):
                        ah[row].fill(a, offset=(rm * rows + row) * (m // 8) * 2112, sizes=[1, K // 256, m // 8, 2112], strides=[0, (M // 8) * 2112, 2112, 1], group=tg)
                pending.append(tg)
                if len(pending) == 2:
                    pending.pop(0).finish()
            for tg in pending:
                tg.finish()

    a_ty = np.ndarray[(M // 8 * (K // 256) * 2112,), np.dtype[np.int8]]
    b_ty = np.ndarray[(N * (K // 512) * 196,), np.dtype[np.uint8]]
    c_ty = np.ndarray[(M * N,), np.dtype[BF16]]
    rt = Runtime(sequence, [a_ty, b_ty, c_ty, [a[0].prod() for a in afs], [b.prod() for b in bhosts], [c.cons() for c in chosts]])
    program = Program(iron.get_current_device(), rt, workers=workers)
    if trace_size:
        events = common.CoreEvent
        program.enable_trace(trace_size, workers=[workers[trace_worker]], egress_shim_col=trace_worker % cols // group_cols, coretile_events=[events.INSTR_EVENT_0, events.INSTR_EVENT_1, events.MEMORY_STALL, events.STREAM_STALL, events.LOCK_STALL, events.INSTR_VECTOR, events.INSTR_CALL, events.INSTR_RETURN])
    return program.resolve_program()
