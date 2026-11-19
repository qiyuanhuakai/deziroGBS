"""Recover a pair-source optical description from a zero-mean pure Bargmann matrix."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter
import networkx as nx
import mpmath as mp
from scipy.linalg import polar
from iteration96_group_common_v1 import ROOT,np,H,save,manifest,Native,D,T,J,B,P,I,U,bind
from iteration96_group_benchmark_v1 import BUILD

def identify(matrix,tolerance=1e-12):
    b=np.asarray(matrix,complex);m=len(b)
    if b.ndim!=2 or b.shape!=(m,m) or not np.isfinite(b).all() or np.max(abs(b-b.T))>tolerance:return None
    graph=nx.from_numpy_array(abs(b)>tolerance)
    signal=[];idler=[];columns=[]
    for cc in nx.connected_components(graph):
        group=graph.subgraph(cc)
        if len(cc)==1 and group.number_of_edges()==0:continue
        if not nx.is_bipartite(group):return None
        colors=nx.bipartite.color(group)
        sides=[sorted(i for i,color in colors.items() if color==v) for v in (0,1)]
        if len(sides[0])!=len(sides[1]):return None
        trials=[]
        for s,h in (sides,sides[::-1]):
            c=b[np.ix_(s,h)];t=np.linalg.norm(c,axis=0)
            if np.any(t==0) or np.any(t>=1):continue
            v,_=polar(c/t)
            trials.append((np.linalg.norm(c-v*t),s,h,v,t))
        if not trials:return None
        _,s,h,v,t=min(trials,key=lambda x:x[0])
        signal.extend(s);idler.extend(h);columns.append((v,t))
    if not columns:return None
    from scipy.linalg import block_diag
    v=block_diag(*(x[0] for x in columns));t=np.concatenate([x[1] for x in columns])
    model=np.zeros_like(b);model[np.ix_(signal,idler)]=v*t;model[np.ix_(idler,signal)]=(v*t).T
    err=float(np.linalg.norm(model-b))
    if err>tolerance:return None
    return np.asarray(signal),np.asarray(idler),v,t,model,err

def physical_distance_bound(b,c):
    # Pure Gaussian vacuum overlaps, recomputed in 65 decimal digits.
    mp.mp.dps=65
    b,c=(mp.matrix(x.tolist()) for x in (b,c));eye=mp.eye(b.rows)
    fidelity=mp.sqrt(mp.det(eye-b.H*b).real*mp.det(eye-c.H*c).real)/abs(mp.det(eye-b.H*c))
    assert 0<=fidelity<=1+mp.mpf('1e-58')
    return float(mp.sqrt(max(0,1-fidelity)))

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'results/iteration96_optical_v1';hfile=ROOT/'results/iteration96_herald_v1/herald.dll'
    files=[Path(__file__),hfile,BUILD,*(src/f'{s}_reference.npz' for s in ['twins','weak','mixed','pure4'])]
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},permutation_seed=96070000,
        scope='pure undisplaced input; certify fitted optical state by determinant overlap before assigning the heralded family'))
    runtime=Native(BUILD);lib=ct.CDLL(str(hfile));prep=bind(lib,'herald96_prepare',[I,D,D,T],P)
    free=bind(lib,'herald96_free',[P],None);ordered=bind(lib,'herald96_ordered',[P,T,J,I],ct.c_double)
    draw=bind(lib,'herald96_draw',[P,I,U,U,I,T,B]);rng=np.random.default_rng(96070000);rows=[]
    for j,name in enumerate(['twins','weak','mixed','unequal','vacuum_padding','pure4','contaminated']):
        original='mixed' if name in ['unequal','vacuum_padding','contaminated'] else name
        data=np.load(src/f'{original}_reference.npz');b=data['barg'].copy()
        if name=='unequal':
            b[:8,8:]*=np.linspace(.5,1.5,8);b[8:,:8]=b[:8,8:].T
        if name=='vacuum_padding':b=np.pad(b,((0,1),(0,1)))
        if name=='contaminated':b[0,0]+=.01
        ix=rng.permutation(len(b));b=b[np.ix_(ix,ix)];start=perf_counter();fit=identify(b)
        row=dict(case=name,eligible=fit is not None,recognition_s=perf_counter()-start)
        if fit is not None:
            s,h,v,t,model,err=fit;d=len(t);dist=physical_distance_bound(b,model)
            assert dist<1e-12;flag=np.ones(1,np.uint32);ptr=prep(d,np.ascontiguousarray(v).view(float),t*t,flag);assert ptr
            try:
                x=np.zeros((2048,2*d),np.uint32);flags=np.zeros(len(x),np.uint8)
                assert draw(ptr,len(x),96070000+j,0,1,x,flags)==0 and not flags.any()
                physical=np.zeros((len(x),len(b)),np.uint32);physical[:,s]=x[:,:d];physical[:,h]=x[:,d:]
                physical=physical[:,np.argsort(ix)]
                assert np.all(physical[:,:8].sum(1)==physical[:,8:16].sum(1))
                if name=='vacuum_padding':assert not physical[:,16].any()
                np.savez_compressed(out/f'{name}.npz',permutation=ix,signal=s,idler=h,v=v,t=t,first=physical[:32])
            finally:free(ptr)
            row.update(phase_sensitive_frobenius=err,optical_trace_distance=dist,active=2*d)
        assert row['eligible']==(name not in ['pure4','contaminated'])
        rows.append(row)
    # Two indistinguishable photons at a balanced beamsplitter cannot leave by different exits.
    v=np.array([[1,1],[1,-1]],complex)/np.sqrt(2);flag=np.ones(1,np.uint32)
    h=prep(2,v.view(float),np.full(2,.25),flag);assert h
    try:
        p=[]
        for seq in [(0,0),(0,1),(1,0),(1,1)]:
            p.append(ordered(h,np.ones(2,np.uint32),np.array(seq,np.int32),2))
        assert np.max(abs(np.array(p)-[.5,0,0,.5]))<1e-14
    finally:free(h)
    v[0,0]=1.1;assert not prep(2,v.view(float),np.full(2,.25),flag)
    save(out,'summary.json',dict(inputs=rows,two_photon_interference=p));manifest(out)
if __name__=='__main__':main()
