"""Train spatial occupancy words on the first chronological half of Jiuzhang 3.0 data."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import H,HIST
from iteration103_gpu_v1 import build as build_gpu,I,U,CU,P
from iteration106_field_v1 import actual_input,DATA
from iteration106_sample_v1 import benchmark
from iteration109_model_v1 import build
from iteration110_device_v1 import config
from iteration110_spectrum_v1 import Spectrum
from iteration110_profile_v1 import Profile


class Stats:
    def __init__(self,groups):
        self.groups = groups
        self.windows = np.zeros((len(groups),256),np.int64)
        self.joint = np.zeros((144,144),np.int64)
        self.total = np.zeros(1153,np.int64)
        self.head = np.zeros(256,np.int64)
        self.records = 0

    def add(self,on,counts,head):
        self.records += len(on)
        f = on.astype(np.float32)
        self.joint += (f.T@f).astype(np.int64)
        self.total += np.bincount(counts,minlength=1153)
        self.head += np.bincount(head,minlength=256)
        for group,h in zip(self.groups,self.windows):
            ids = group[group>=0]
            word = on[:,ids]@(1 << np.arange(len(ids)))
            h += np.bincount(word,minlength=256)


class Windows(Profile):
    def add_windows(self,dll,groups,cdf):
        self.windows_lib = ct.CDLL(str(dll))
        self.apply_windows = self.windows_lib.window_emit111
        self.apply_windows.argtypes = [I,I,I,U,CU,P,P,P,P,P]
        self.gpu_groups,self.gpu_cdf = [torch.as_tensor(np.ascontiguousarray(a),device='cuda') for a in (groups,cdf)]
        self.move = None
        self.without_windows = self.emit
        self.emit = self.emit_windows

    def emit_windows(self,g,f,s,n,pn,wide,seed,off,field,eig,lut,cdf,out,error,stream):
        rc = self.without_windows(g,f,s,n,pn,wide,seed,off,field,eig,lut,cdf,out,error,stream)
        if rc:return rc
        return self.apply_windows(n,len(self.gpu_groups),wide,seed,off,
                                  self.gpu_groups.data_ptr(),self.gpu_cdf.data_ptr(),self.move.data_ptr() if self.move is not None else 0,out,stream)


def compare(v,ref,opt):
    p,q = v.joint/v.records,ref.joint/ref.records
    delta = p-q
    singles = np.diag(delta)
    both = .5*(abs(delta)+abs(singles[:,None]-delta)+abs(singles[None,:]-delta)+abs(delta-singles[:,None]-singles[None,:]))
    pair = both[np.triu_indices(144,1)]
    wtv = abs(v.windows/v.records-ref.windows/ref.records).sum(1)/2
    return dict(single_mean_abs=float(abs(singles).mean()),single_max_abs=float(abs(singles).max()),
        pair_mean_tv=float(pair.mean()),pair_max_tv=float(pair.max()),anchor_pair_tv=float(both[36,107]),
        window_mean_tv=float(wtv.mean()),window_max_tv=float(wtv.max()),
        total_ks=float(abs((v.total/v.records-ref.total/ref.records).cumsum()).max()),
        first8_hw_tv=float(abs(v.head/v.records-ref.head/ref.records).sum()/2),
        first8_optical_tv=float(abs(v.head/v.records-opt).sum()/2))


def run(out,raw_path,redraw=False):
    out.mkdir(parents=True,exist_ok=True)
    build_gpu(out,'spatial_windows111_v1.cu')
    torch.backends.cuda.matmul.allow_tf32=False
    z = np.load(DATA/'iteration110_joint_v1/tables.npz')
    frozen = sorted({0,34,35,*map(int,z['anchor']//8),*map(int,z['direct']//8)})
    ids = [j for i in range(72) for j in (i,143-i) if j not in frozen]
    groups = np.pad(ids,(0,(-len(ids))%8),constant_values=-1).astype(np.int32).reshape(-1,8)
    # Keep the named opposing sites together, independently of holdout outcomes.
    i,j = np.argwhere(groups==36)[0],np.argwhere(groups==107)[0]
    if i[0]!=j[0]:
        other=(i[0],0 if i[1] else 1)
        groups[tuple(j)],groups[other]=groups[other],groups[tuple(j)]
    files = json.loads((HIST.parent/'plan.json').read_text())['hashes']
    path,sha = next((p,h) for p,h in files.items() if p.endswith('high power (1).bin'))
    path = raw_path or Path(path)
    assert H(path)==sha
    raw = np.memmap(path,np.uint8,mode='r').reshape(-1,144)
    profile = DATA/'iteration110_profile_v1'
    bank = np.load(profile/'training.npz')
    train_total = bank['train_total']
    cut = int(train_total.sum())
    train,hold = Stats(groups),Stats(groups)
    pop = np.array([i.bit_count() for i in range(256)],np.uint8)
    reverse = np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1)@(1<<np.arange(8))
    for first,last,stats in ((0,cut,train),(cut,len(raw),hold)):
        for off in range(first,last,4096):
            word=raw[off:min(last,off+4096)]
            stats.add(word!=0,pop[word].sum(1).astype(int),reverse[word[:,0]])
    assert np.array_equal(train.total,train_total)
    cond = np.stack([train.windows*(pop==k) for k in range(9)],axis=1).cumsum(2).astype(np.uint32)
    np.savez_compressed(out/'training.npz',groups=groups,cdf=cond,sha256=sha)
    t,r = actual_input()
    m=Spectrum(build(t,r,8,1)).materialize(z['coeff'])
    prior=np.load(DATA/'iteration110_device_v1/histograms.npz')
    weights=abs(t)**2@(np.sinh(r)**2).sum(0).repeat(2)
    args=(m,config(z,'fitted'),weights,DATA/'iteration110_device_v1/gpu.dll',DATA/'iteration110_hardware_v1/gpu.dll',
          [prior[f'fitted_{i}_totals'] for i in (0,1)],train_total)
    orig,eng=Profile(*args),Windows(*args)
    for model in (orig,eng):
        model.attach(profile/'gpu.dll',bank['profiles'],bank['frequency'].cumsum(dtype=np.uint32),bank['starts'])
    eng.add_windows(out/'gpu.dll',groups,cond)
    opt=np.load(DATA/'iteration109_device_v1/observed.npz')['first8_target']
    arrays=dict(groups=groups,optical=opt)
    def freeze(name,stats):
        for field in ('joint','windows','total','head'):arrays[name+'_'+field]=getattr(stats,field)
    freeze('hold',hold);freeze('train',train)
    rows=[]
    for pn in (0,1):
        seed=111400000+pn
        pilot=np.zeros_like(train.windows)
        for off in range(0,262144,4096):
            rec=orig.draw(4096,pn,seed+500,off).reshape(-1,144,8).any(2)
            for group,h in zip(groups,pilot):
                ids=group[group>=0]
                h+=np.bincount(rec[:,ids]@(1<<np.arange(len(ids))),minlength=256)
        gates=np.zeros_like(pilot,dtype=np.uint64)
        deficit=np.zeros_like(cond)
        for k in range(9):
            use=pop==k
            src=pilot[:,use]
            tgt=train.windows[:,use]
            p=src/np.maximum(1,src.sum(1,keepdims=True))
            q=tgt/np.maximum(1,tgt.sum(1,keepdims=True))
            delta=np.where((tgt.sum(1)>0)[:,None],q-p,0)
            gates[:,use]=np.rint(2**52*np.maximum(-delta,0)/np.where(p>0,p,1)).astype(np.uint64)
            freq=np.maximum(delta,0)
            coeff=freq/np.maximum(1e-30,freq.sum(1,keepdims=True))
            deficit[:,k,use]=np.rint(coeff*2**30).astype(np.uint32)
        repaired=deficit.cumsum(2,dtype=np.uint32)
        arrays[f'pilot{pn}']=pilot
        arrays[f'gate{pn}']=gates
        arrays[f'deficit{pn}']=repaired
        eng.move=None if redraw else torch.as_tensor(gates,device='cuda')
        eng.gpu_cdf=torch.as_tensor(cond if redraw else repaired,device='cuda')
        times=[benchmark(lambda n,s,o:eng.draw(n,pn,s,o),n,seed+100,7) for n in (2048,16384)]
        before,after=Stats(groups),Stats(groups)
        shots=131072
        changed=0
        for off in range(0,shots,2048):
            a=orig.draw(2048,pn,seed,off).copy()
            b=eng.draw(2048,pn,seed,off)
            aa=a.reshape(-1,144,8);bb=b.reshape(-1,144,8)
            assert np.array_equal(aa[:,frozen],bb[:,frozen])
            assert np.array_equal(a.sum(1),b.sum(1))
            for j in range(1,int(max(a.max(),b.max()))+1):
                assert np.array_equal((a==j).sum(1),(b==j).sum(1))
            active=aa.any(2)!=bb.any(2)
            changed+=sum(np.any(active[:,group[group>=0]],axis=1).sum() for group in groups)
            for data,stats in ((aa,before),(bb,after)):
                hits=data>0
                stats.add(hits.any(2),hits.sum((1,2)).astype(int),hits[:,0]@(1<<np.arange(8)))
        assert np.array_equal(before.total,after.total) and np.array_equal(before.head,after.head)
        freeze(f'pre{pn}',before);freeze(f'post{pn}',after)
        rows.append(dict(pnr=bool(pn),shots=shots,timing=times,changed_fraction=float(changed/(shots*len(groups))),
                         before=compare(before,hold,opt),after=compare(after,hold,opt)))
        print(pn,rows[-1]['before']['pair_mean_tv'],rows[-1]['after']['pair_mean_tv'],flush=True)
    np.savez_compressed(out/'holdout.npz',**arrays)
    (out/'windows.json').write_text(json.dumps(dict(train=cut,holdout=len(raw)-cut,frozen=frozen,
            split_reference=compare(train,hold,opt),routes=rows),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    ap.add_argument('--raw',type=Path)
    ap.add_argument('--redraw',action='store_true')
    args=ap.parse_args()
    with threadpool_limits(limits=3):
        run(args.output.resolve(),args.raw,args.redraw)
