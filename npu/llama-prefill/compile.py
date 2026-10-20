from pathlib import Path
import aie.iron as iron
from aie.iron.device import from_name
from design import matmul
iron.set_current_device(from_name('npu2', n_cols=8))
Path('artifacts').mkdir()
for M, m in [(64,16),(512,128)]:
    design=matmul.specialize(M=M,K=2560,N=640,m=m,am=16,n=16,cols=8,rows=4,cache_a=False,a_depth=2,stack_size=4096,l2_depth=2,trace_size=0,trace_worker=0)
    design.compile(xclbin_path=str(Path(f'artifacts/m{M}.xclbin').resolve()),inst_path=str(Path(f'artifacts/m{M}.bin').resolve()))
