"""Independent eight-site words on the two new captures, exact optical references and 8-bit Frugal."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration112_hardware_v1 import DATA,engines,attach,actual_input
from iteration107_cliques_v1 import optical_x
from iteration107_fast_v1 import gaussian_click,oracle,ONE


def spatial_reference(t,r,sites):
    tt=t[(sites[:,None]*8+np.arange(8)).ravel()]
    x=optical_x(tt,r)
    v=np.ones(256)
    for m in range(1,256):
        ids=(np.flatnonzero(m&(1<<np.arange(8)))[:,None]*8+np.arange(8)).ravel()
        ix=np.r_[ids,ids+64]
        sig,logdet=np.linalg.slogdet(np.eye(len(ix))+x[:,ix][:,:,ix])
        assert np.all(sig==1)
        v[m]=np.exp(-logdet.sum()/2)
    p=v[::-1].copy()
    for j in range(8):
        a=p.reshape(-1,2,1<<j);a[:,1]-=a[:,0]
    assert p.min()>-1e-12 and abs(p.sum()-1)<1e-10
    return p


def hist(values):
    codes=values@(1<<np.arange(8))
    return np.array([np.bincount(k,minlength=256) for k in codes.T])


def run(out,captures):
    out.mkdir(exist_ok=True,parents=True)
    rng=np.random.default_rng(112842021)
    groups=rng.permutation(144).reshape(18,8)
    ids=8*groups+rng.integers(0,8,groups.shape)
    arrays=dict(groups=groups,ids=ids)
    t,r=actual_input();quad=oracle(True)
    spatial=np.array([spatial_reference(t,r,g) for g in groups])
    matrices=[optical_x(t[p],r) for p in ids]
    detector=np.array([gaussian_click(quad,x,ONE) for x in matrices])
    arrays['spatial_optical']=spatial;arrays['detector_optical']=detector
    for item in json.loads((captures/'files.json').read_text()):
        data=np.memmap(captures/item['member'],np.uint8,mode='r').reshape(-1,144)
        bits=np.zeros((18,256),int);banks=bits.copy()
        for off in range(0,len(data),8192):
            a=data[off:off+8192,groups]
            banks+=hist(a!=0);bits+=hist(((a>>(7-ids%8))&1).astype(np.uint8))
        arrays[f'file{item["file"]}_spatial']=banks;arrays[f'file{item["file"]}_detector']=bits
    make=engines();cal=np.load(DATA/'iteration112_hardware_v1/calibration.npz')
    for pn in (0,1):
        e=make(pn)
        for phase in (0,1):
            if phase:attach(e,cal['groups'],cal[f'gate{pn}'],cal[f'deficit{pn}'])
            bits=np.zeros((18,256),int);banks=bits.copy()
            for off in range(0,262144,2048):
                a=e.draw(2048,pn,112843000+pn,off)>0
                banks+=hist(a.reshape(-1,144,8).any(2)[:,groups])
                bits+=hist(a[:,ids])
                if off+2048==65536:arrays[f'equal_r{pn}_{phase}_detector']=bits.copy()
            arrays[f'r{pn}_{phase}_spatial']=banks;arrays[f'r{pn}_{phase}_detector']=bits
    rt=os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
    cpu=ct.CDLL(str(DATA/'iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll'))
    P=ct.c_void_p;I=ct.c_int
    cpu.fc73_sample.argtypes=[I,I,I,ct.c_uint,I]+[P]*4+[I]
    msg=ct.create_string_buffer(4096);timings=[]
    for k in (2,4):
        h=np.zeros((18,256),int);us=[]
        for i,xx in enumerate(matrices):
            cov=np.ascontiguousarray(np.eye(16)+2*xx)
            for rep in range(32):
                n=2048;st=np.zeros(7);a=np.empty((2,n,8),np.uint8)
                start=perf_counter()
                for j in (0,1):
                    rc=cpu.fc73_sample(8,k,n,112850000+10000*k+100*i+2*rep+j,1,cov[j].ctypes.data,a[j].ctypes.data,st.ctypes.data,msg,len(msg))
                    assert rc==0 and st[5]==64,msg.value
                record=a[0]|a[1]
                if rep>=2:us.append((perf_counter()-start)*1e6/n)
                h[i]+=np.bincount(record@(1<<np.arange(8)),minlength=256)
        arrays[f'frugal{k}']=h
        timings.append(dict(k=k,shots_per_panel=int(h[0].sum()),median_us=float(np.median(us)),quartiles=np.quantile(us,[.25,.75]).tolist()))
    def tv(arr,ref):
        p=arr/arr.sum(1,keepdims=True);q=ref/ref.sum(1,keepdims=True)
        v=abs(p-q).sum(1)/2
        return dict(mean=float(v.mean()),maximum=float(v.max()),per_panel=v.tolist())
    rows=[]
    for kind,target in [('spatial',spatial),('detector',detector)]:
        for method in [f'r{i}_{j}' for i in (0,1) for j in (0,1)]+(['equal_r0_1','equal_r1_1','frugal2','frugal4'] if kind=='detector' else []):
            h=arrays[method if method.startswith('frugal') else f'{method}_{kind}']
            rows.append(dict(kind=kind,method=method,shots_per_panel=int(h[0].sum()),optical=tv(h,target),
                        file10=tv(h,arrays['file10_'+kind]),file11=tv(h,arrays['file11_'+kind])))
        rows.append(dict(kind=kind,method='hardware10_vs11',comparison=tv(arrays['file10_'+kind],arrays['file11_'+kind])))
    np.savez_compressed(out/'panels.npz',**arrays)
    (out/'panels.json').write_text(json.dumps(dict(results=rows,frugal_timing=timings),indent=2)+'\n',encoding='utf8')
    print([(v['kind'],v['method'],v['file10']['mean']) for v in rows if 'file10' in v],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    ap.add_argument('--captures',type=Path,default=DATA.parent.parent/'external_sources/jiuzhang3_iteration112_independent')
    args=ap.parse_args()
    with threadpool_limits(limits=2):run(args.output.resolve(),args.captures.resolve())
