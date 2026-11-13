"""Two independent 5M-record laws; simultaneous PNR and threshold quality bounds.
8 one-sided nominal bounds share failure probability .01. Pilot-fixed events
provide TV lower bounds; convexity and bounded differences give upper bounds.
Ideal iid and numerical target assumptions are explicit, not PRNG certification.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,math,itertools
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_sphere_number_pnr import NativeSphereNumberPNRSampler
from gbslab.native_transport_alias_pnr import NativeTransportAliasPNRSampler
from gbslab.native_resplit_pnr import NativeResplitPNRSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
models={'transport':(NativeTransportAliasPNRSampler,ROOT/'results/iteration80_transport_pilot_v3/transport.dll'),'sphere':(NativeSphereNumberPNRSampler,ROOT/'results/iteration78_pilot_v2/sphere.dll')};runtime=[Path(shutil.which('g++')).parent];N=5000000;batch_size=250000;radius=math.sqrt(math.log(800)/(2*N));panels=[(1,'transport',8),(1,'sphere',8)];seed_base=0xac000000000000;pilot_base=0xad000000000000;stride=2**40
save('plan.json',dict(scope=__doc__,driver_sha256=H(Path(__file__)),binary_hashes={kind:H(p) for kind,(_,p) in models.items()},calibration_sha256=H(ROOT/'results/iteration80_ipf_pilot_v1/strong4.npz'),panels=panels,samples_per_law=N,batch_size=batch_size,pilot_count=200000,seed_base=seed_base,pilot_base=pilot_base,stride=stride,one_sided_bounds=8,joint_failure=.01,radius=radius));shutil.copy2(__file__,OUT/Path(__file__).name);rows=[]
for panel,(ci,kind,k) in enumerate(panels):
    data=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{ci}.npz');v,t,cutoff,codes,target=data['v'],data['t'],int(data['cutoff']),data['codes'],data['target'];m=len(v);base=cutoff+1;powers=np.array([base**j for j in range(m)],np.uint64);lookup=np.full(base**m,-1,np.int32);lookup[codes]=np.arange(len(codes),dtype=np.int32);clickpowers=np.array([2**j for j in range(m)],np.uint64)
    o=np.block([[v.real,-v.imag],[v.imag,v.real]]);x=o@np.diag(np.r_[-t/(1+t),t/(1-t)])@o.T;vacuum=[]
    for mask in range(2**m):
        ids=[j for j in range(m) if mask>>j&1];ids+= [j+m for j in ids];vacuum.append(float(1/np.sqrt(np.linalg.det(np.eye(len(ids))+x[np.ix_(ids,ids)]))))
    clicktarget=np.zeros(2**m)
    for mask in range(2**m):
        zero=(2**m-1)^mask;sub=mask
        while True:
            clicktarget[mask]+=(-1)**sub.bit_count()*vacuum[zero|sub]
            if sub==0:break
            sub=(sub-1)&mask
    assert clicktarget.min()>=0 and abs(clicktarget.sum()-1)<1e-12
    def histograms(counts):
        total=counts.sum(axis=1);inside=total<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];assert labels.min()>=0
        return np.bincount(labels,minlength=len(target)),np.bincount(((counts>0).astype(np.uint64)@clickpowers).astype(np.int64),minlength=2**m)
    cls,binary=models[kind]
    table=np.load(ROOT/'results/iteration80_ipf_pilot_v1/strong4.npz');extra=dict(couplings=table['couplings'],latent_weights=table['weights']) if kind=='transport' else {}
    with cls(v,t,binary,workers=16,runtime_dirs=runtime,exact_total_budget=k,**extra) as model:
        pilot=model.sample(200000,pilot_base+panel*stride)['counts'];hp,hc=histograms(pilot);ep=hp/200000>target;ec=hc/200000>clicktarget;np.savez_compressed(OUT/f'pilot_{panel}.npz',counts=pilot,pnr_event=ep,click_event=ec,click_target=clicktarget);hist=np.zeros(len(target),np.int64);clickhist=np.zeros(2**m,np.int64);seconds=0.;odd=photons=maximum=0;statrows=[]
        for batch in range(20):
            seed=seed_base+(panel*20+batch)*stride;save(f'started_{panel}_{batch}.json',dict(seed=seed,count=batch_size));r=model.sample(batch_size,seed);counts=r['counts'];a,b=histograms(counts);hist+=a;clickhist+=b;total=counts.sum(axis=1);photons+=int(total.sum());odd+=int(np.count_nonzero(total%2));maximum=max(maximum,int(total.max()));seconds+=r['wall_seconds'];statrows.append(r['statistics']);np.savez_compressed(OUT/f'counts_{panel}_{batch}.npz',counts=counts);save(f'status_{panel}_{batch}.json',dict(seed=seed,seconds=r['wall_seconds'],statistics=r['statistics']));print('batch',panel,batch,r['wall_seconds'],flush=True)
    def bounds(hist,target,event,tail):
        empirical=hist/N;tv=float(abs(empirical-target).sum()/2);lower=max(0.,float((empirical[event]-target[event]).sum())-radius);upper=min(1.,tv+radius+tail);return dict(empirical_tv=tv,lower=lower,upper=upper)
    row=dict(parameter_name='exact_total_budget',tail_records=sum(s.get('tail_records',0) for s in statrows),panel=panel,case=ci,kind=kind,retained=k,records=N,pnr=bounds(hist,target,ep,target[-1]),click=bounds(clickhist,clicktarget,ec,0.),odd_records=odd,maximum_photons=maximum,mean_photons=photons/N,seconds=seconds,target_tail=float(target[-1]),target_mean=float(np.sum(t*t/(1-t*t))),resplit_attempts=sum(s.get('resplit_attempts',0) for s in statrows),resplit_successes=sum(s.get('resplit_successes',0) for s in statrows));rows.append(row);save(f'row_{panel}.json',row);np.savez_compressed(OUT/f'histogram_{panel}.npz',pnr=hist,click=clickhist);print(row,flush=True)
save('summary.json',dict(passed=True,rows=rows,radius=radius));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
