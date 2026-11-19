"""Time single physical records from the same fixed third-moment ensemble."""
import ctypes as C
import argparse
from iteration92_common_v2 import ROOT,Path,np,os,shutil,time,save,manifest,H

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
base=ROOT/'results/iteration95_third_click_v1'
layout=ROOT/'results/iteration83_loss_pilot_v1/input.npz'
raws=[ROOT/f'results/iteration94_actual_quality_v1/input_{s}.npz' for s in (0,1)]
files=[Path(__file__),base/'third.dll',base/'quality_3.npz',layout,*raws]
save(out,'plan.json',dict(experiment='0x95b20000',records=101,first='warmup',
    hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}))
runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
lib=C.CDLL(str(base/'third.dll'));I,P,U=C.c_int,C.c_void_p,C.c_uint64
D,B=[np.ctypeslib.ndpointer(dtype=x,flags='C_CONTIGUOUS') for x in (float,np.uint8)]
def bind(name,args,rest=I):
    f=getattr(lib,name);f.argtypes=args;f.restype=rest;return f
mom=bind('window95_moments',[I,I,D,D,D,D,D,D])
prep=bind('window95_prepare',[I,I,I,I,D,D,D],P)
tilt=bind('window95_tilt',[P,I,C.c_double],P)
make=bind('ensemble95_prepare',[I,C.POINTER(P),D],P)
draw=bind('ensemble95_sample',[P,I,U,U,I,B])
destroy=bind('window95_destroy',[P],None);free=bind('ensemble95_destroy',[P],None)
m=1152;q=np.empty((2,2*m,2*m));mean=np.empty((2,2*m))
for i,path in enumerate(raws):
    z=np.load(path);u=np.ascontiguousarray(z['u'])
    assert mom(m,len(z['t']),u.view(float),z['t'],z['alpha'].view(float),z['noise'],q[i],mean[i])==0
order=np.argsort(np.load(layout)['order']);ix=np.r_[order,order+m]
window=prep(m,4,4,2,np.ascontiguousarray(q[:,ix][:,:,ix]),np.ascontiguousarray(mean[:,ix]),np.zeros(4))
assert window
calibration=np.load(base/'quality_3.npz')
components=[tilt(window,4,a) for a in calibration['slopes']]
assert all(components)
handle=make(9,(P*9)(*components),calibration['weights']);assert handle
control=np.empty((32,m),np.uint8)
assert draw(handle,32,0x95b00003,20,1,control)==0
assert np.array_equal(np.packbits(control,axis=1,bitorder='little'),calibration['packed_bits'][20:52])
seconds=[];records=[]
for i in range(101):
    now=time.perf_counter()
    bits=np.empty((1,m),np.uint8);rc=draw(handle,1,0x95b20000,i,1,bits)
    elapsed=time.perf_counter()-now
    assert rc==0
    if i:seconds.append(elapsed);records.append(bits)
for c in components+[window]:destroy(c)
free(handle)
np.savez_compressed(out/'single.npz',seconds=seconds,bits=np.packbits(np.concatenate(records),axis=1,bitorder='little'))
save(out,'summary.json',dict(warm_single_median_seconds=float(np.median(seconds)),repetitions=100))
manifest(out)
