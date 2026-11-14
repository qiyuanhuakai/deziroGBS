"""Fresh domain-separated coherent-core PNR and click quality, frozen48-event family."""
import argparse,math
from fractions import Fraction as F
from iteration92_common_v2 import *

def confidence(labels,target,N,pilot,delta,tail=0.):
    h=np.bincount(labels,minlength=len(target));ph=np.bincount(labels[:pilot],minlength=len(target));vh=np.bincount(labels[pilot:],minlength=len(target));event=ph/pilot>target;tv=float(abs(h/N-target).sum()/2);gap=float(vh[event].sum()/(N-pilot)-target[event].sum());return dict(empirical_tv=tv,lower=max(0.,gap-math.sqrt(math.log(1/delta)/(2*(N-pilot)))),upper=min(1.,tv+math.sqrt(math.log(1/delta)/(2*N))+tail)),dict(histogram=h,pilot=ph,validation=vh,event=event,target=target)

def reference(case):
    if 'reference88' in case:
        j=case['reference88'];z=np.load(ROOT/f'results/iteration88_small_pnr_reference_v1/case_{j}.npz');meta=read(ROOT/f'results/iteration88_small_pnr_reference_v1/case_{j}.json');p=z['target'].ravel();tail=float(meta['tail_upper']);target=np.r_[p,float(meta['tail_mass']),0.];u=np.eye(2,dtype=complex)
        if case['eta']<1:u=np.vstack([math.sqrt(case['eta'])*u,math.sqrt(1-case['eta'])*u])
        return u,target,np.r_[z['target_click'],0.],tail,None
    j=case['reference91'];k=case['k'];base=ROOT/'results/iteration91_fock_quality_v1';u=np.load(base/f'input_{j}.npz')['u'];z=np.load(base/f'law_{j}_{k}.npz');order=np.argsort(z['keys']);keys=z['keys'][order];p=z['probabilities'][order];s=F(case['r'])**2;tail=1-sum((F(math.comb(k+l,k))*(1-s)**(k+1)*s**l for l in range((case['cutoff']-k)//2+1)),F(0));target=np.r_[p,float(tail),0.];clicklabels=np.zeros(len(keys),int)
    for m in range(len(u)):clicklabels|=(((keys>>(8*m))&255)>0).astype(int)<<m
    # This finite click reference omits tail. Assign it to vacuum only for a
    # normalized reference; account for its full TV cost in both bounds.
    click=np.r_[np.bincount(clicklabels,weights=p,minlength=1<<len(u)),0.];click[0]+=float(tail)
    return u,target,click,float(tail),keys

def labels_for(counts,errors,case,keys,target_length):
    m=case['detected_modes'];counts=counts[:,:m]
    if keys is None:
        inside=(counts<=64).all(1);labels=np.full(len(counts),65**2,int);labels[inside]=counts[inside,0]*65+counts[inside,1]
    else:
        packed=np.zeros(len(counts),np.uint64)
        for j in range(m):packed|=counts[:,j].astype(np.uint64)<<(8*j)
        pos=np.searchsorted(keys,packed);inside=pos<len(keys);inside[inside]&=keys[pos[inside]]==packed[inside];labels=np.full(len(counts),len(keys),int);labels[inside]=pos[inside]
    labels[errors!=0]=target_length-1;cl=np.zeros(len(counts),int)
    for j in range(m):cl|=(counts[:,j]>0).astype(int)<<j
    cl[errors!=0]=1<<m;return labels,cl

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);planpath=ROOT/'research/iteration92_quality_plan_v1.json';plan=read(planpath);assert plan['experiment_base']==0x92002000;lib,runtime=load();files=[Path(__file__),planpath,ROOT/'experiments/iteration92_common_v2.py',ROOT/'native/gaussian_coherent_core92_v2.cpp',ROOT/'native/gaussian_rng_domain84_v1.hpp',ROOT/'results/iteration92_build_v2/core.dll'];save(out,'plan.json',dict(quality=plan,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}));assert read(ROOT/'results/iteration92_gates_v2/validation.json')['passed'];N=plan['records'];pilot=plan['pilot_records'];delta=plan['confidence']['failure_budget']/plan['confidence']['events'];rows=[]
    for j,case in enumerate(plan['cases']):
        u,target,ctarget,tail,keys=reference(case);c=fock_core(case['k']);np.savez_compressed(out/f'input_{j}.npz',u=u,core=c,target=target,click_target=ctarget);raws=[];metas=[]
        for offset in range(0,N,plan['batch_size']):
            raw,meta=sample(lib,u,c,case['r'],min(plan['batch_size'],N-offset),plan['experiment_base']+j,offset=offset,threads=plan['threads'],cap=plan['photon_cap']);np.savez_compressed(out/f'raw_{j}_{offset}.npz',**raw);save(out,f'row_{j}_{offset}.json',meta);raws.append(raw);metas.append(meta)
            if offset%16384==0:print('quality',j,offset,meta,flush=True)
        raw={key:np.concatenate([r[key] for r in raws]) for key in raws[0]};labels,cl=labels_for(raw['counts'],raw['errors'],case,keys,len(target));pnr,ph=confidence(labels,target,N,pilot,delta,tail);click,ch=confidence(cl,ctarget,N,pilot,delta,tail);click['lower']=max(0,click['lower']-tail);np.savez_compressed(out/f'hist_{j}.npz',**{'pnr_'+k:v for k,v in ph.items()},**{'click_'+k:v for k,v in ch.items()});row=dict(case=case,experiment=plan['experiment_base']+j,records=N,failures=int(np.count_nonzero(raw['errors'])),numerical_failures=int(np.count_nonzero(raw['errors']==2)),seconds=sum(x['seconds'] for x in metas),pnr=pnr,click=click,target_tail=tail,mean_photons=float(raw['counts'][:,:case['detected_modes']].sum(1).mean()),maximum_core_degree=float(raw['statistics'][:,2].max()),maximum_root_residual=float(raw['statistics'][:,4].max()),maximum_hazard_residual=float(raw['statistics'][:,7].max()));rows.append(row);save(out,f'quality_{j}.json',row);print('finished',row,flush=True)
    old=[];oldbase=ROOT/'results/iteration88_small_pnr_quality_v1'
    for row in read(oldbase/'summary.json')['rows']:
        j=row['case_index'];depth=row['retained'];z=np.load(oldbase/f'combined_{j}_{depth}.npz');case=dict(reference88=j,detected_modes=2,eta=1);_,target,_,tail,_=reference(case);labels,_=labels_for(z['counts'],~z['valid'],case,None,len(target));q,_=confidence(labels,target,len(labels),4096,delta,tail);old.append(dict(case=row['case'],retained=depth,records=len(labels),pnr=q))
    save(out,'summary.json',dict(completed=True,rows=rows,historical88=old,confidence=plan['confidence'],scope=__doc__));manifest(out)
