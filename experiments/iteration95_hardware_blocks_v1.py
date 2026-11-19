"""Test count-distribution homogeneity across 16 consecutive stored portions of the hardware file."""
import argparse
import math
from iteration92_common_v2 import ROOT,Path,np,C,os,shutil,H,save,manifest

p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
ref=ROOT/'results/iteration81_jiuzhang_accuracy_v3'
model=ROOT/'results/iteration95_third_click_v1/quality_3.npz'
raw=ROOT.parent/'external_sources/jiuzhang3_iteration81_sample_v1/extracted_3/jiuzhang3/data/high power/high power (1).bin'
size=raw.stat().st_size
assert size%144==0
n=size//144;groups=16
edges=np.array([i*n//groups for i in range(groups+1)])
save(out,'plan.json',dict(raw_sha256=H(raw),records=n,edges=edges.tolist(),family_error=.01,
    metric='Each contiguous stored-file block versus its disjoint complement: count-CDF sup-distance; Bonferroni over 32 one-sample DKW events. Null is a common iid count distribution.',
    purpose='Evidence for or against homogeneity in stored order; rejection can also be caused by file organization. No assumption of timestamps.',
    files={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),ref/'hist.dll',ref/'histograms.npz',model]}))
runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
lib=C.CDLL(str(ref/'hist.dll'));fn=lib.jz81_hist
B=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
I=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
U=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
fn.argtypes=[B,C.c_uint64,C.c_int,I,C.c_uint64,C.c_int,U,U]
reference=np.load(ref/'histograms.npz')
ports=np.ascontiguousarray(reference['ports'],dtype=np.int32)
data=np.memmap(raw,dtype=np.uint8,mode='r',shape=(n,144))
h=np.empty((groups,1153),np.uint64);marg=np.empty((groups,3,2,256),np.uint64)
for i in range(groups):
    chunk=data[edges[i]:edges[i+1]]
    assert fn(chunk,len(chunk),3,ports,0,4,marg[i],h[i])==0 and int(h[i].sum())==len(chunk)
totals=h.sum(0)
assert np.array_equal(totals,reference['total_click_histogram'])
assert np.array_equal(marg.sum(axis=(0,2)),reference['histograms'].sum(1))
rows=[];cdf=np.load(model)['count'].cumsum();x=np.arange(1153);delta=.01/(groups*2)
for i,row in enumerate(h):
    count=int(row.sum());complement=totals-row
    local=row.cumsum()/count
    difference=abs(local-complement.cumsum()/(n-count))
    threshold=sum(math.sqrt(math.log(2/delta)/(2*k)) for k in (count,n-count))
    rows.append(dict(block=i,n=count,mean=float(row@x/count),sup=float(max(difference)),threshold=threshold,
                     rejected=bool(max(difference)>threshold),model_ks=float(max(abs(local-cdf)))))
np.savez_compressed(out/'histograms.npz',counts=h,patterns=marg,edges=edges)
save(out,'summary.json',dict(rejections=sum(r['rejected'] for r in rows),rows=rows));manifest(out)
