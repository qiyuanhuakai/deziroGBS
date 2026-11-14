"""Independent saved-data verification of all111 frozen confidence events.
No sampler, experiment analysis helper or native DLL is imported. Rebuild
histograms, failure bounds, domain intervals, timing medians and file hashes.
Address non-reuse and scoped numerical checks are not a proof of true iid or
an interval-certified bound on all floating-point implementation errors.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.special import betaincinv
ROOT=Path(__file__).resolve().parents[1];checks=0
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def H(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(ok,label):
    global checks
    checks+=1
    if not ok:raise AssertionError(label)
def near(a,b,label):check(abs(float(a)-float(b))<3e-12,label)
def save(p,value):p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def bounds(labels,target,delta,pilot,tail=0):
    n=len(labels);h=np.bincount(labels,minlength=len(target));ph=np.bincount(labels[:pilot],minlength=len(target));vh=np.bincount(labels[pilot:],minlength=len(target));event=ph/pilot>target;distance=np.abs(h/n-target).sum()/2;gap=vh[event].sum()/(n-pilot)-target[event].sum();return dict(empirical_tv=float(distance),lower=max(0.,float(gap)-math.sqrt(math.log(1/delta)/(2*(n-pilot)))),upper=min(1.,float(distance)+math.sqrt(math.log(1/delta)/(2*n))+tail))
def compare(result,stored,label):
    for key,value in result.items():near(value,stored[key],label+':'+key)
def pfail(k,n,delta):return 1. if k==n else float(betaincinv(k+1,n-k,1-delta))
def philox_word(experiment,sector,purpose,row,draw):
    mask=(1<<32)-1;block=draw//2;c=[block&mask,block>>32,row,(sector<<16)|purpose];k=[experiment&mask,experiment>>32]
    for round in range(10):
        a=0xd2511f53*c[0];b=0xcd9e8d57*c[2];c=[(b>>32)^c[1]^k[0],b&mask,(a>>32)^c[3]^k[1],a&mask]
        if round<9:k=[(k[0]+0x9e3779b9)&mask,(k[1]+0xbb67ae85)&mask]
    lane=2*(draw%2);return c[lane]|(c[lane+1]<<32)

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);plan=read(ROOT/'research/iteration93_quality_plan_v1.json');delta=plan['confidence']['failure_budget']/plan['confidence']['events'];check(len(plan['confidence']['event_names'])==len(set(plan['confidence']['event_names']))==111,'frozen event family');save(out/'plan.json',dict(scope=__doc__,source_sha256=H(Path(__file__)),quality_plan_sha256=H(ROOT/'research/iteration93_quality_plan_v1.json')))
for folder in sorted((ROOT/'results').glob('iteration93*')):
    for path in folder.rglob('manifest.json'):
        entries=read(path);entries=entries.get('artifacts',entries)
        for name,h in entries.items():check(H(path.parent/name)==h,'manifest '+str(path.parent/name))
    for path in folder.rglob('plan.json'):
        for name,h in read(path).get('hashes',{}).items():check(H(ROOT/name)==h,'plan hash '+name)
for name,h in plan['hashes'].items():check(H(ROOT/name)==h,'quality frozen hash '+name)
for file in ('iteration93_gates_v5/validation.json','iteration93_dense_audit_v6/validation.json','iteration93_center_audit_v1/validation.json','iteration93_reference_v2/validation.json','iteration93_rng_batch_build_v1/validation.json','iteration93_baseline_pilot_v1/validation.json','iteration93_baseline_provenance_v2/validation.json','iteration93_upstream_check_v1/validation.json'):
    check(read(ROOT/'results'/file)['passed'],'gate '+file)
suite=read(ROOT/'results/iteration93_tests_v1/validation.json');check(suite['passed'] and suite['mps93_passed']==10,'full suite with10 MPS tests')
for name,h in read(ROOT/'results/iteration93_tests_v1/source_hashes.json').items():check(H(ROOT/name)==h,'suite hash '+name)
domains={};actual_domains=0;seeds={};pcg_states=set();rows=[];events=[]
def register(experiment,sector,purpose,offset,length,used):
    global actual_domains
    key=(experiment,sector,purpose);intervals=domains.setdefault(key,[]);check(0<=experiment<2**64 and 0<=sector<2**16 and 0<purpose<2**16 and 0<=offset<2**32 and offset+length<=2**32,'domain bounds');check(all(offset>=b or offset+length<=a for a,b in intervals),'domain intervals disjoint');intervals.append((offset,offset+length));check(np.isfinite(used).all() and (used>=0).all() and (used==np.floor(used)).all() and (used<2**53).all(),'exact saved usage counters');actual_domains+=int(np.count_nonzero(used))

for job in plan['jobs']:
    family=job['family'];folder=ROOT/f"results/iteration93_{family}_quality_v1"/job['name'];stored=read(folder/'quality.json');j=job['case'];ref=np.load(ROOT/f'results/iteration93_reference_v2/law_{j}.npz');refmeta=read(ROOT/f'results/iteration93_reference_v2/row_{j}.json');parts=[];failures=[];statparts=[];total_seconds=0
    for offset in range(0,job['records'],job['batch_size']):
        data=np.load(folder/f'raw_{offset}.npz');meta=read(folder/f'row_{offset}.json');key='counts' if 'counts' in data else 'clicks';counts=data[key];failed=data['failed'];n=len(counts);check(n==min(job['batch_size'],job['records']-offset),'fixed sample count');check(failed.dtype==bool and len(failed)==n,'failure array');check(np.isfinite(counts).all(),'finite records');parts.append(counts);failures.append(failed);total_seconds+=meta.get('wall_seconds_including_rng',meta.get('seconds',0));check(meta['failures']==int(failed.sum()),'batch failure ledger')
        if family=='mps':
            s=data['statistics'];check(np.array_equal(counts.sum(1),s[:,0]),'MPS photon counter');check(np.array_equal(~failed,s[:,1].astype(bool)),'MPS completion counter');check(np.array_equal(failed,data['errors']!=0),'MPS errors');check(np.isin(data['errors'],[0,1,2]).all(),'no hidden numerical failures');check((s[:,5]<=job['budget']).all() and (s[:,5]>=0).all(),'charged budgets');check((s[:,8]<=job['bond']).all(),'retained ranks');check((s[:,14]==s[:,0]+1).all(),'waiting draw count');check((s[:,15]>=s[:,0]).all(),'detector draw count');statparts.append(s)
            for purpose,column in ((1,14),(2,15),(3,16)):register(job['experiment'],0,purpose,offset,n,s[:,column])
            if j in (0,2):check(not (counts[~failed].sum(1)%2).any(),'pure even parity')
        elif family=='old':
            s=data['statistics'];check(np.array_equal(~failed,data['valid']),'old failure retention');check(np.array_equal(counts.sum(1),s[:,0]),'old photon counter')
            for purpose,column in ((1,12),(2,13),(3,14),(4,15)):register(job['experiment'],0,purpose,offset,n,s[:,column])
        elif family=='direct':
            draws=data['draws'];check(draws.shape==counts.shape,'direct full draw shape');register(job['experiment'],0,5,offset,n,np.full(n,counts.shape[1]));check(np.array_equal(failed,(data['errors'][:,0]!=0)|(data['statistics'][:,5]!=counts.shape[1])),'direct failed rows')
            for row in (0,n-1):
                for column in (0,counts.shape[1]-1):near(draws[row,column],(philox_word(job['experiment'],0,5,offset+row,column)>>11)*2**-53,'independent Philox draw')
        else:
            expected=0x93040000+job['id']*1024+offset//job['batch_size'];check(meta['seed']==expected and expected not in seeds,'third party batch seed');seeds[expected]=(family,job['name'],offset)
            if family=='pe':
                state=np.random.default_rng(expected).bit_generator.state;pair=(state['state']['state'],state['state']['inc']);check(pair not in pcg_states,'PE distinct actual PCG initialization');pcg_states.add(pair);check(not failed.any(),'PE no censored rows');check(np.isfinite(data['round_off']).all(),'PE numerical diagnostics finite')
    counts=np.concatenate(parts);failed=np.concatenate(failures);N=len(counts);m=counts.shape[1];check(N==job['records'],'total fixed records');check(stored['failures']==int(failed.sum()),'all failures kept');cl=sum((counts[:,i]>0).astype(int)<<i for i in range(m));cl[failed]=1<<m;click=bounds(cl,np.r_[ref['click'],0.],delta,plan['pilot_records']);compare(click,stored['click'],'click');events+=['click_lower','click_upper'];result=dict(name=job['name'],family=family,case=j,records=N,failures=int(failed.sum()),click=click)
    if family in ('mps','old','pe'):
        if j==2:
            values=counts.sum(1).astype(np.int64);values[failed]=2**31;ecdf=np.array([(values<=k).mean() for k in range(65)]);ks=float(np.max(abs(ecdf-np.cumsum(ref['total']))));radius=math.sqrt(math.log(2/delta)/(2*N));cdf=dict(empirical_distance=ks,lower=max(0.,ks-radius),upper=min(1.,ks+radius+float(refmeta['tail_chernoff'])),radius=radius);compare(cdf,stored['total_cdf'],'total CDF');result['total_cdf']=cdf;events+=['total_cdf']
        else:
            if j<2:
                keys=ref['keys'];packed=sum(counts[:,i].astype(np.uint64)<<(8*i) for i in range(m));position=np.searchsorted(keys,packed);inside=position<len(keys);inside[inside]&=keys[position[inside]]==packed[inside];labels=np.full(N,len(keys));labels[inside]=position[inside];target=np.r_[ref['probabilities'],float(refmeta['total_tail']),0.]
            else:
                inside=(counts<=64).all(1);labels=np.full(N,65**2);labels[inside]=65*counts[inside,0]+counts[inside,1];target=np.r_[ref['pnr'].ravel(),float(refmeta['pnr_tail']),0.]
            labels[failed]=len(target)-1;check(abs(target.sum()-1)<2e-12 and (target>=0).all(),'normalized PNR reference');pnr=bounds(labels,target,delta,plan['pilot_records'],float(refmeta['tail_chernoff']));compare(pnr,stored['pnr'],'PNR');result['pnr']=pnr;events+=['pnr_lower','pnr_upper']
    if family=='mps':
        s=np.concatenate(statparts);pf=pfail(int(failed.sum()),N,delta);near(pf,stored['failure_probability_upper'],'Clopper Pearson upper');near(min(1.,pf+job['budget']),stored['ideal_arithmetic_compression_tv_upper'],'ideal model bound');near(s[:,5].max(),stored['maximum_charged'],'maximum charge');result.update(failure_upper=pf,ideal_arithmetic_tv_upper=min(1.,pf+job['budget']));events+=['failure_upper']
    near(total_seconds,stored['seconds'],'recorded batch times');rows.append(result)
actual=plan['actual'];folder=ROOT/'results/iteration93_actual_quality_v1';sectors=[]
for sector in (0,1):
    parts=[];bad=[]
    for offset in range(0,actual['records'],actual['batch_size']):
        z=np.load(folder/f'raw_{sector}_{offset}.npz');s=z['statistics'];check(len(s)==actual['batch_size'],'actual fixed batch');check(np.array_equal(z['counts'].sum(1),s[:,0]),'actual photon counter');check((s[:,5]<=actual['budgets'][sector]).all(),'actual budget');parts.append(z['counts']);bad.append(z['errors']!=0)
        for purpose,column in ((1,14),(2,15),(3,16)):register(actual['experiment'],sector,purpose,offset,len(s),s[:,column])
    sectors.append((np.concatenate(parts),np.concatenate(bad)))
combined=sectors[0][0]+sectors[1][0];failed=sectors[0][1]|sectors[1][1];stored=read(folder/'summary.json');check(stored['completed'] and int(failed.sum())==stored['failures'],'actual completion');z=np.load(folder/'combined.npz');check(np.array_equal(z['counts'],combined) and np.array_equal(z['failed'],failed),'actual merge');values=combined.sum(1).astype(np.int64);values[failed]=2**31;reference=np.load(ROOT/'results/iteration84_photon_cdf_v1/reference.npz')['photon_cdf'];ecdf=np.array([(values<=k).mean() for k in range(len(reference))]);ks=float(np.max(abs(ecdf-reference)));radius=math.sqrt(math.log(2/delta)/(2*len(values)));pf=pfail(int(failed.sum()),len(values),delta);near(ks,stored['cdf']['empirical_distance'],'actual KS');near(radius,stored['cdf']['radius'],'actual DKW');near(pf,stored['failure_upper'],'actual failure upper');near(min(1.,sum(actual['budgets'])+pf),stored['ideal_arithmetic_compression_tv_upper'],'actual ideal TV bound');events+=['actual CDF','actual failure'];check(len(events)==111,'reconstructed all111 events')
timing=read(ROOT/'results/iteration93_timing_v1/summary.json');check(timing['completed'],'timing completed')
for row in timing['rows']:
    name=row['job']['timing_name'];observed=[]
    for rep in range(3):
        raw=read(ROOT/f'results/iteration93_timing_v1/row_{rep}_{name}.json');observed.append(raw['seconds']);z=np.load(ROOT/f'results/iteration93_timing_v1/raw_{rep}_{name}.npz');check(int(z['failed'].sum())==raw['failures'],'timing failures')
    near(np.median(observed),row['median_seconds'],'timing median');check(np.array_equal(observed,row['seconds']),'timing observations')
worker=read(ROOT/'results/iteration93_rng_batch_build_v1/frugal_worker_seeds.json');check(len(worker)==224 and len({r['worker_seed'] for r in worker})==224,'actual Frugal worker seeds');check(all(r['external_seed'] in seeds and seeds[r['external_seed']][0]=='frugal' for r in worker),'Frugal worker provenance')
result=dict(passed=True,checks=checks,confidence_events=len(events),quality_records=sum(r['records'] for r in rows)+actual['records'],native_used_domains=actual_domains,third_party_batch_seeds=len(seeds),pe_distinct_pcg_states=len(pcg_states),frugal_distinct_worker_seeds=len(worker),rows=rows,actual=stored,timings=timing['rows'],tests=suite,scope=__doc__);save(out/'validation.json',result);save(out/'manifest.json',{p.name:H(p) for p in out.iterdir() if p.is_file()});print(json.dumps(dict(passed=True,checks=checks,quality_records=result['quality_records'],domains=actual_domains)))
