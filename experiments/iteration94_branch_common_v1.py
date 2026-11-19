"""Bindings for direct Gaussian-projection branch sampling and diagnostic trees."""
from iteration94_common_v2 import *
U64P=C.POINTER(C.c_uint64)

def gaussian94(z):
    u=z['u'];t=z['t'];alpha=z['alpha'];noise=z['noise'];m,n=u.shape
    real=np.block([[u.real,-u.imag],[u.imag,u.real]])
    x=np.diag(np.r_[t/(1-t),-t/(1+t)])+2*noise@noise.T
    return np.ascontiguousarray(np.eye(2*m)+real@x@real.T),np.ascontiguousarray(np.sqrt(2)*np.r_[(u@alpha).real,(u@alpha).imag])

def load_branch94():
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));lib=C.CDLL(str(ROOT/'results/iteration94_branch_build_v1/branch.dll'));return lib,runtime

def branch_sample94(lib,q,mean,N,experiment,limit=8,budget=.01,sector=0,offset=0,threads=1):
    q=np.ascontiguousarray(q,float);mean=np.ascontiguousarray(mean,float);m=len(mean)//2;bits=np.zeros((N,m),np.uint32);stats=np.zeros((N,16));errors=np.zeros(N,np.int32);error=C.create_string_buffer(4096)
    fn=lib.branch94_sample;fn.argtypes=[C.c_int,DP,DP,C.c_int,C.c_double,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,UP,DP,IP,C.c_char_p,C.c_int]
    start=time.perf_counter();rc=fn(m,ptr(q),ptr(mean),limit,budget,N,experiment,sector,offset,threads,bits.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),error,len(error));seconds=time.perf_counter()-start
    return dict(clicks=bits,statistics=stats,errors=errors),dict(returncode=rc,error=error.value.decode(),seconds=seconds,failures=int(np.count_nonzero(errors)),codes={str(k):int(np.count_nonzero(errors==k)) for k in (1,2,3,4)})

def branch_enumerate94(lib,q,mean,limit,budget):
    q=np.ascontiguousarray(q,float);mean=np.ascontiguousarray(mean,float);m=len(mean)//2;law=np.zeros((1<<m)+1);stats=np.zeros(5);error=C.create_string_buffer(4096);fn=lib.branch94_enumerate;fn.argtypes=[C.c_int,DP,DP,C.c_int,C.c_double,DP,DP,C.c_char_p,C.c_int]
    start=time.perf_counter();rc=fn(m,ptr(q),ptr(mean),limit,budget,ptr(law),ptr(stats),error,len(error));assert rc==0,error.value
    return law,stats,time.perf_counter()-start

def branch_compress94(lib,q,mean,masks,coefficients,limit):
    q=np.ascontiguousarray(q,float);mean=np.ascontiguousarray(mean,float);m=len(mean)//2;masks=np.ascontiguousarray(masks,np.uint64);coefficients=np.ascontiguousarray(coefficients,float);n=len(coefficients);new_masks=np.zeros((limit,(m+63)//64),np.uint64);new_coefficients=np.zeros(limit);stats=np.zeros(16);error=C.create_string_buffer(4096)
    fn=lib.branch94_compress_probe;fn.argtypes=[C.c_int,DP,DP,C.c_int,U64P,DP,C.c_int,DP,U64P,DP,C.c_char_p,C.c_int]
    rc=fn(m,ptr(q),ptr(mean),n,masks.ctypes.data_as(U64P),ptr(coefficients),limit,ptr(new_coefficients),new_masks.ctypes.data_as(U64P),ptr(stats),error,len(error));return dict(returncode=rc,error=error.value.decode(),coefficients=new_coefficients,masks=new_masks,statistics=stats)
