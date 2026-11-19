"""Test whether keeping only the ranks needed by the local tail target lowers cost."""
import argparse
from iteration94_common_v2 import ROOT,Path,np,C,DP,UP,IP,ptr,H,save,manifest,load93,time

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
prior=ROOT/'results/iteration95_adaptive_pnr_v1'
paths=[ROOT/f'results/iteration94_actual_quality_v1/input_{s}.npz' for s in (0,1)]
small=ROOT/'results/iteration93_reference_v2/input_0.npz'
save(out,'plan.json',dict(old_paired_records=8,new_paired_records=4,
    arms={'fixed32':[32,32,6],'adaptive32':[32,48,8],'adaptive4':[4,48,8]},
    scope='Identical accepted-state charge accounting, local target 0.0006; budget 0.1 and 0.01. Only rank floor differs between adaptive arms.',
    seeds=['0x95a00000 recovered comparison, new floor','0x95a02000 all arms paired, all records retained'],
    small='Four-mode dense reference, 16384 new records for rank-floor 1 versus 4, same cap 16 and site limit 8. 0x95a03000.',
    hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*paths,small,prior/'adaptive.dll',prior/'manifest.json']}))
legacy,runtime=load93();lib=C.CDLL(str(prior/'adaptive.dll'))
set_lapack=lib.mps93_set_lapack;set_lapack.argtypes=[C.c_void_p]*3
assert set_lapack(*[C.cast(getattr(legacy._blas,f),C.c_void_p) for f in ('scipy_LAPACKE_zgesdd','scipy_LAPACKE_zgeqrf','scipy_LAPACKE_zungqr')])==0
fn=lib.mps95_sample;I=C.c_int
fn.argtypes=[I,I,DP,DP,DP,DP,I,I,C.c_double,I,I,C.c_uint64,C.c_uint,C.c_uint64,I,UP,DP,IP,C.c_char_p,I,I,I,C.c_double,DP]
def run(z,exp,sector,off,n,base_d,base_rank,cap_d,cap_rank,budget,cap):
    u,t,alpha,noise=[np.ascontiguousarray(z[k]) for k in ('u','t','alpha','noise')]
    output=np.empty((n,len(u)),np.uint32);stats=np.empty((n,18));errors=np.empty(n,np.int32)
    tr=np.empty((n,cap,14));msg=C.create_string_buffer(4096)
    now=time.perf_counter()
    rc=fn(len(u),len(t),ptr(u.view(float)),ptr(t),ptr(alpha.view(float)),ptr(noise),base_d,base_rank,budget,cap,n,exp,
        sector,off,min(4,n),output.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),msg,len(msg),cap_d,cap_rank,.0006,ptr(tr))
    elapsed=time.perf_counter()-now
    assert rc==int(any(errors)) and np.isin(errors,[0,1,2,3,4]).all(),msg.value
    assert np.array_equal(output.sum(1),stats[:,0]) and np.all(stats[:,5]<=budget)
    for row,e in enumerate(errors):
        length=int(stats[row,0])+int(e==2)
        log=tr[row,:length]
        assert np.all(log[:,8]<=cap_rank) and np.all(log[:,12]<=cap_d)
        assert abs(log[:int(stats[row,0]),10].sum()-stats[row,5])<1e-11
        assert np.max(abs(log[:,10]**2-log[:,3]**2-(1-log[:,3]**2)*log[:,4]**2),initial=0)<1e-13
    return dict(counts=output,errors=errors,statistics=stats,trace=tr,message=np.array(msg.value.decode())),elapsed
first=ROOT/'results/iteration95_rank_floor_v1'
import json
old_rows=json.loads((first/'rows.json').read_text())
save(out,'recovery.json',dict(reason='v1 stopped before serializing the second batch when a native per-record status fell outside {0,2}. All native statuses now retained. Re-execute only the unsaved batch.',
    first_hashes={p.name:H(p) for p in first.iterdir() if p.is_file()}))
rows=[]
arms={'fixed32':(32,32,6),'adaptive32':(32,48,8),'adaptive4':(4,48,8)}
for series,n,exp in [('prior',8,0x95a00000),('new',4,0x95a02000)]:
    for off in range(0,n,4):
        for sector in (0,1):
            for name in (['adaptive4'] if series=='prior' else (['adaptive4','fixed32','adaptive32'] if not sector else ['adaptive32','fixed32','adaptive4'])):
                floor,ceiling,levels=arms[name]
                if series=='prior' and off==0:
                    saved=first/f'prior_0_{sector}_adaptive4.npz'
                    z=dict(np.load(saved))
                    seconds=next(r['seconds'] for r in old_rows if r['sector']==sector)
                else:
                    z,seconds=run(dict(np.load(paths[sector])),exp,sector,off,4,6,floor,levels,ceiling,(.1,.01)[sector],256)
                np.savez_compressed(out/f'{series}_{off}_{sector}_{name}.npz',**z)
                rows.append(dict(series=series,offset=off,sector=sector,arm=name,seconds=seconds,errors=z['errors'].tolist()))
                save(out,'rows.json',rows)
                print(series,off,sector,name,z['errors'].tolist(),round(seconds,2),flush=True)
for floor in (1,4):
    c=[];e=[];seconds=0
    for off in range(0,16384,256):
        raw,wall=run(dict(np.load(small)),0x95a03000,0,off,256,4,floor,8,16,.02,64)
        c.append(raw['counts']);e.append(raw['errors']);seconds+=wall
    np.savez_compressed(out/f'small_{floor}.npz',counts=np.concatenate(c),errors=np.concatenate(e))
    save(out,f'small_{floor}.json',dict(seconds=seconds,failures=int(np.count_nonzero(np.concatenate(e)))))
# Replay exceptional statuses in isolation to recover each native diagnostic message.
for row in rows:
    for i,code in enumerate(row['errors']):
        if code not in (0,2):
            floor,ceiling,levels=arms[row['arm']]
            sec=row['sector'];idx=row['offset']+i
            exp=0x95a00000 if row['series']=='prior' else 0x95a02000
            replay,_=run(dict(np.load(paths[sec])),exp,sec,idx,1,6,floor,levels,ceiling,(.1,.01)[sec],256)
            assert replay['errors'][0]==code
            save(out,f"diagnostic_{row['series']}_{idx}_{sec}_{row['arm']}.json",dict(code=code,message=str(replay['message'])))
manifest(out)
