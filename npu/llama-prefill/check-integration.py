import os,ctypes,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import reference as ref
root=Path(os.environ['LLAMA_TEST_ROOT'])
os.environ['LLAMA_BASE_LIB']=str(root/'lib/libggml-base.so')
class Init(ctypes.Structure):
 _fields_=[('mem_size',ctypes.c_size_t),('mem_buffer',ctypes.c_void_p),('no_alloc',ctypes.c_bool)]
ptr=ctypes.c_void_p;size=ctypes.c_size_t;i64=ctypes.c_int64
lib=ctypes.CDLL(str(root/'lib/libggml-base.so'),mode=ctypes.RTLD_GLOBAL);vk=ctypes.CDLL(str(root/'bin/libggml-vulkan.so'))
def bind(l,n,r,*a):
 f=getattr(l,n);f.restype=r;f.argtypes=list(a);return f
init=bind(lib,'ggml_init',ptr,Init);t3=bind(lib,'ggml_new_tensor_3d',ptr,ptr,ctypes.c_int,i64,i64,i64);t2=bind(lib,'ggml_new_tensor_2d',ptr,ptr,ctypes.c_int,i64,i64)
view2=bind(lib,'ggml_view_2d',ptr,ptr,ptr,i64,i64,size,size)
mm=bind(lib,'ggml_mul_mat_id',ptr,ptr,ptr,ptr,ptr);scale=bind(lib,'ggml_scale',ptr,ptr,ptr,ctypes.c_float)
newgraph=bind(lib,'ggml_new_graph_custom',ptr,ptr,size,ctypes.c_bool);expand=bind(lib,'ggml_build_forward_expand',None,ptr,ptr)
alloc=bind(lib,'ggml_backend_alloc_ctx_tensors',ptr,ptr,ptr);put=bind(lib,'ggml_backend_tensor_set',None,ptr,ptr,size,size);get=bind(lib,'ggml_backend_tensor_get',None,ptr,ptr,size,size)
compute=bind(lib,'ggml_backend_graph_compute',ctypes.c_int,ptr,ptr);sync=bind(lib,'ggml_backend_synchronize',None,ptr)
backend=bind(vk,'ggml_backend_vk_init',ptr,size)(0);assert backend
mode=sys.argv[1] if len(sys.argv)>1 else 'hybrid'
M=1 if mode=='decode' else 512;K,N=2560,640;slots=2
if os.environ.get('LLAMA_NPU_TEST_WEIGHTS'):
 q0=np.load(os.environ['LLAMA_NPU_TEST_WEIGHTS'])
else:
 q0=np.random.default_rng(9).integers(0,256,size=(640,10,98),dtype=np.uint8)
 q0[...,:2]=np.array([2.0**-10],dtype='<f2').view(np.uint8)
q=np.stack([q0,np.roll(q0,13,axis=0),q0[::-1].copy()])
x=np.random.default_rng(400).normal(size=(M,2,K)).astype(np.float32)
ids=np.empty((M,slots),dtype=np.int32);ids[:,0]=0;ids[:,1]=1+np.arange(M)%2
ctx=init(Init(1024*1024,None,True));w=t3(ctx,18,K,N,3);a=t3(ctx,0,K,2,M);ip=t2(ctx,26,slots+2,M);it=view2(ctx,ip,slots,M,(slots+2)*4,0)
# Put an operation before and after the routed matmul to verify graph splitting.
a2=scale(ctx,a,.5);out=scale(ctx,mm(ctx,w,a2,it),2.)
graph=newgraph(ctx,64,False);expand(graph,out);buf=alloc(ctx,backend);assert buf
put(w,q.ctypes.data,0,q.nbytes);put(a,x.ctypes.data,0,x.nbytes);padded_ids=np.full((M,slots+2),-999,dtype=np.int32);padded_ids[:,:slots]=ids
put(ip,padded_ids.ctypes.data,0,padded_ids.nbytes)
def run(enabled):
 os.environ['GGML_VK_NPU_PREFILL']='1' if enabled else '0'
 assert compute(backend,graph)==0;sync(backend)
 y=np.empty((M,slots,N),dtype=np.float32);get(out,y.ctypes.data,0,y.nbytes);return y
baseline=run(False);actual=run(True)
np.testing.assert_allclose(actual[:,1],baseline[:,1],rtol=.001,atol=.0001)
if mode in ('decode','fallback','missing'):
 np.testing.assert_allclose(actual,baseline,rtol=.001,atol=.0001)
else:
 _,wc,ws=ref.decode(q0);_,xc,xs=ref.pack_activations(x[:,0]*.5);expected=ref.matmul(xc,xs,wc,ws)*2
 np.testing.assert_array_equal(actual[:,0],expected)
# A second call with changed activations must not reuse stale rows.
x*=-1;put(a,x.ctypes.data,0,x.nbytes);changed=run(True)
baseline_changed=run(False)
if mode == 'hybrid':
 np.testing.assert_array_equal(changed[:,0],-actual[:,0])
 np.testing.assert_allclose(changed[:,1],baseline_changed[:,1],rtol=.001,atol=.0001)
else:
 np.testing.assert_allclose(changed,baseline_changed,rtol=.001,atol=.0001)
zero_weights=np.zeros_like(q);put(w,zero_weights.ctypes.data,0,zero_weights.nbytes)
np.testing.assert_array_equal(run(True),np.zeros_like(actual))
result=dict(strided_expert_ids_checked=True,changed_weights_checked=True,mode=mode,passed=True,shape=[M,slots,K,N],relative_l2_vs_vulkan=float(np.linalg.norm(actual-baseline)/max(np.linalg.norm(baseline),1e-30)),gpu_remainder_matches=True,changed_activations_checked=True)
(Path(os.environ.get('LLAMA_NPU_TEST_RESULTS','work'))/('test-llama-npu-'+mode+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(result)
bind(lib,'ggml_backend_buffer_free',None,ptr)(buf);bind(lib,'ggml_free',None,ptr)(ctx);bind(lib,'ggml_backend_free',None,ptr)(backend)
