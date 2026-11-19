"""Independent exact density law, exact physical covariance and CDF replay."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode,exact_determinant
from iteration45_independent import inverse,integer_density_law,grid_below,mobius,tv
from iteration46_displacement_independent import exp_inverse_positive,density_vacuum_coefficient
from iteration47_radicals import density_radicals
from iteration48_independent import independent_overlap,bisect_sqrt
from gbslab.local_mps import threshold_step,sample_local_threshold
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def verify_bridge(rec,tensors,alpha,check,tag,out,index):
    job=rec['job'];n=job['modes'];d=job['dimension'];inp=rec['input'];r=inp['real'];im=inp['imag'];a=F(1,2) if job['strength']=='moderate' else F(3,4);zero=F(0)
    if job['family']=='single':er=[[a]];ei=[[zero]]
    elif job['family']=='paired':er=[[zero,a],[a,zero]];ei=[[zero,zero],[zero,zero]]
    else:
        b=F(1,4) if job['strength']=='moderate' else F(1,2);er=[[(a-b)/2,zero],[zero,(b-a)/2]];ei=[[zero,(a+b)/2],[(a+b)/2,zero]]
    check(tag+'predeclared exact input',r==er and im==ei and len(r)==n and inp['family']==job['family'] and all(t.shape[1]==d for t in tensors));expected=np.array(([0,0] if job['displacement']==0 else [.5+.25j,-.25+.5j])[:n],complex);check(tag+'predeclared displacement',np.array_equal(alpha,expected));br=rec['bridge'];check(tag+'box dimensions',br['dimensions']==[d]*n and br['occupations']==[list(x) for x in np.ndindex(*([d]*n))]);ind=independent_overlap(tensors,r,im);save(out/f'case_{index:02d}.overlap.json',encode(ind));check(tag+'independent polynomial coefficients',encode(ind['hermite'])==encode(br['hermite']));check(tag+'projection coefficient norm',ind['projection_unnormalized_norm']==br['projection_unnormalized_norm']);scales=[max(F(float(z)).denominator for z in np.concatenate([t.real.ravel(),t.imag.ravel()])) for t in tensors];scale=math.prod(scales);check(tag+'independent rational MPS amplitudes',all([ar*scale,ai*scale]==v for (ar,ai),v in zip(ind['amplitudes'],br['integer_amplitudes'])) and ind['norm']*scale**2==br['integer_norm'])
    S=np.block([[np.array(r,dtype=object),np.array(im,dtype=object)],[np.array(im,dtype=object),-np.array(r,dtype=object)]]);A=np.eye(2*n,dtype=object)-S;det=exact_determinant(A.tolist());inv=np.array(inverse(A.tolist()),dtype=object);V=2*inv-np.eye(2*n,dtype=object);omega=np.block([[np.zeros((n,n),dtype=object),np.eye(n,dtype=object)],[-np.eye(n,dtype=object),np.zeros((n,n),dtype=object)]]);check(tag+'exact Gaussian covariance and purity',V.tolist()==br['physical']['covariance'] and det==br['physical']['determinant'] and np.array_equal(V@omega@V,omega));check(tag+'physical positive leading minors',all(exact_determinant(A[:k,:k].tolist())>0 for k in range(1,2*n+1)));normlo,normhi=bisect_sqrt(det);l,h=br['normalization_probability_interval'];check(tag+'independent Gaussian normalization',l<=normlo<=normhi<=h)
    overlap=ind['overlap_squared_over_norm_interval'];lo,hi=br['overlap_squared_interval'];check(tag+'independent density overlap',lo/br['integer_norm']<=overlap[0]<=overlap[1]<=hi/br['integer_norm']);R=ind['projection_unnormalized_norm'];pairs=[('projection_mass_interval',(normlo*R,normhi*R)),('projected_fidelity_interval',(overlap[0]/R,overlap[1]/R)),('full_fidelity_interval',(normlo*overlap[0],normhi*overlap[1]))]
    for name,(lo,hi) in pairs:
        lo=max(F(0),lo);hi=min(F(1),hi);l,h=br[name];check(tag+'independent '+name,l<=lo<=hi<=h)
    for fid,dist in [('projection_mass_interval','tail_trace_interval'),('projected_fidelity_interval','compression_trace_interval'),('full_fidelity_interval','full_trace_interval')]:
        fl,fh=br[fid];dl,dh=br[dist];check(tag+'trace square inequalities '+dist,0<=dl<=dh<=1 and dl*dl<=1-fh<=1-fl<=dh*dh)
    check(tag+'tail mass identity',br['tail_mass_interval']==[1-br['projection_mass_interval'][1],1-br['projection_mass_interval'][0]])
    if job['family']=='paired':
        analytic=[(F(math.factorial(o[0]))*a**o[0],F(0)) if o[0]==o[1] else (F(0),F(0)) for o in np.ndindex(d,d)];check(tag+'paired exact Hermite control',encode(analytic)==encode(br['hermite']));check(tag+'paired analytic geometric tail',br['tail_mass_interval'][0]<=a**(2*d)<=br['tail_mass_interval'][1])
    if job['family']=='single':
        analytic=[(F(0),F(0)) if k%2 else (F(math.factorial(k),2**(k//2)*math.factorial(k//2))*a**(k//2),F(0)) for k in range(d)];check(tag+'single analytic squeezed coefficients',encode(analytic)==encode(br['hermite']))
    target=rec['gaussian_target'];check(tag+'Gaussian target uses same exact covariance',target['physical']['covariance']==V.tolist());z=[F(float(x.real)) for x in alpha]+[F(float(x.imag)) for x in alpha];vac=[]
    for mask,row in enumerate(target['records']):
        modes=[j for j in range(n) if mask>>j&1];ix=modes+[j+n for j in modes]
        if ix:
            Q=[[(V[i,j]+int(i==j))/2 for j in ix] for i in ix];di=exact_determinant(Q);qi=inverse(Q);exponent=sum(z[i]*qi[j][k]*z[l] for j,i in enumerate(ix) for k,l in enumerate(ix))
        else:di=F(1);exponent=F(0)
        check(tag+'Gaussian determinant/exponent '+str(mask),row['mask']==mask and row['determinant']==di and row['exponent']==exponent);rl,rh=bisect_sqrt(1/di);el,eh=exp_inverse_positive(exponent);l,h=row['factor_interval'];check(tag+'Gaussian independent radical '+str(mask),l<=rl<=rh<=h);l,h=row['exponential_interval'];check(tag+'Gaussian independent exponential '+str(mask),l<=el<=eh<=h);v=target['vacuum_intervals'][mask];check(tag+'Gaussian independent vacuum '+str(mask),v[0]<=rl*el<=rh*eh<=v[1]);vac.append((rl*el,rh*eh))
    raw,law=mobius(vac,n)
    for k,(l,h) in enumerate(law):
        pl,ph=target['intervals'][k];check(tag+'independent full Gaussian atom '+str(k),pl<=l<=h<=ph)
    raw,law=mobius(target['vacuum_intervals'],n);check(tag+'full Gaussian interval Mobius',encode(raw)==encode(target['raw_intervals']) and encode(law)==encode(target['intervals']))


def main():
    out=ROOT/'results/iteration48_gaussian_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration48_gaussian','iteration48_independent','iteration47_radicals','iteration46_displacement_independent','iteration45_independent','iteration22_external','iteration33_run','verify_iteration34')]+[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration48_gaussian';plan=read(base+'/plan.json');summary=read(base+'/summary.json');check('all60complete',summary['planned']==summary['finished']==summary['completed']==60)
    for n,s in plan['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/base/'source_snapshot'/n))
    for index,job in enumerate(plan['jobs']):
        n=job['modes'];tag=f'case{index} ';rec=read(base+f'/case_{index:02d}.json');name=base+f'/case_{index:02d}.npz';deps[name]=digest(ROOT/name);stored=np.load(ROOT/name);alpha=stored['alpha'];tensors=[stored[f'tensor_{j}'] for j in range(n)];check(tag+'job',rec['job']==job);check(tag+'stored C layout',all(a.flags.c_contiguous and list(a.strides)==stride for a,stride in zip(tensors,rec['stored_strides'])));verify_bridge(rec,tensors,alpha,check,tag,out,index);target=rec['finite_target'];independent=density_radicals(tensors,alpha);save(out/f'case_{index:02d}.radicals.json',encode(independent));vac=[]
        for k in range(max(a.shape[1] for a in tensors)):
            a,b=target['factorial_root_intervals'][str(k)];factor=math.factorial(k);check(tag+'factorial root '+str(k),0<a<=b and a*a*factor<=1<=b*b*factor and b-a<=F(1,1<<192))
        for mask,r in enumerate(target['vacuum_records']):
            cl,ch=independent[mask]['coefficient_interval'];a,b=r['coefficient_interval'];check(tag+'independent radical coefficient '+str(mask),a<=cl<=ch<=b);exponent=sum(F(float(alpha[j].real))**2+F(float(alpha[j].imag))**2 for j in range(n) if mask>>j&1);el,eh=exp_inverse_positive(exponent);l,h=r['exponential_interval'];check(tag+'independent exponential '+str(mask),l<=el<=eh<=h and r['exponent']==exponent and r['mask']==mask);raw=(a*l,b*h);value=(max(F(0),raw[0]),min(F(1),raw[1]));check(tag+'vacuum product '+str(mask),list(raw)==r['raw_vacuum_interval'] and value[0]<=cl*el<=ch*eh<=value[1]);vac.append(value)
        check(tag+'vacuum intervals',encode(vac)==encode(target['vacuum_intervals']));raw,law=mobius(vac,n);check(tag+'independent displaced law',encode(raw)==encode(target['raw_intervals']) and encode(law)==encode(target['intervals']));check(tag+'rigorous target width',max(b-a for a,b in law)<F(1,10**40))
        sampling=rec['sampling'];trace={tuple(node['path']):node for node in sampling['trace']};check(tag+'unique nodes',len(trace)==len(sampling['trace'])==sampling['nodes']);real=[F(0)]*((1<<n)+1);word=[F(0)]*((1<<n)+1);queue=[((),0,np.ones(1,complex),F(1),F(1))];seen=set();leafcount=0
        for path,mask,left,rm,wm in queue:
            if len(path)==n:real[mask]+=rm;word[mask]+=wm;leafcount+=1;continue
            node=trace[path];seen.add(path);check(tag+'node '+str(path),node['mask']==mask and node['real_mass']==rm and node['word_mass']==wm and node['left_hex']==[[float(z.real).hex(),float(z.imag).hex()] for z in left]);branches,weights,total=threshold_step(left,tensors[len(path)],alpha[len(path)]);p=weights/total;cdf=p.cumsum();cdf/=cdf[-1];check(tag+'native arithmetic '+str(path),node['status']=='completed' and node['weights_hex']==[float(x).hex() for x in weights] and node['total_hex']==total.hex() and node['probabilities_hex']==[float(x).hex() for x in p] and node['cdf_hex']==[float(x).hex() for x in cdf]);previous=F(0);previous_count=0;counts=[]
            for channel,edge in enumerate(cdf):
                e=F(float(edge));count=grid_below(e,53);width=e-previous;c=count-previous_count;counts.append(c);previous=e;previous_count=count;rchild=rm*width;wchild=wm*F(c,1<<53)
                if not rchild and not wchild:continue
                if weights[channel]<=0:real[-1]+=rchild;word[-1]+=wchild;continue
                queue.append((path+(channel,),mask|((channel!=0)<<len(path)),branches[channel]/math.sqrt(weights[channel]),rchild,wchild))
            check(tag+'independent grid counts '+str(path),counts==node['counts'] and previous==1 and previous_count==1<<53)
        check(tag+'complete sampling law',seen==set(trace) and leafcount==sampling['leaves'] and real==sampling['real_law'] and word==sampling['word_law'] and sum(real)==sum(word)==1)
        cal=rec['calibration'];rng=np.random.default_rng(cal['seed']);outputs,info=sample_local_threshold(tensors,np.tile(alpha,(cal['samples'],1)),rng);check(tag+'public sampler replay',outputs.tolist()==cal['outputs'] and len(cal['records'])==n*cal['samples']);rng=np.random.default_rng(cal['seed']);path=()
        for c in cal['records']:
            if len(path)==n:path=()
            node=trace[path];probs=np.array([float.fromhex(x) for x in node['probabilities_hex']]);channel=int(rng.choice(len(probs),p=probs));check(tag+'recorded choice '+str(len(path)),c['path']==list(path) and c['probabilities_hex']==node['probabilities_hex'] and c['channel']==channel);path+=(channel,)
        value=tv(word,law+[(F(0),F(0))]);check(tag+'measurement TV interval',list(value)==rec['arithmetic_TV']);full=tv(word,rec['gaussian_target']['intervals']+[(F(0),F(0))]);check(tag+'full Gaussian TV',list(full)==rec['full_Gaussian_TV']);upper=min(F(1),rec['bridge']['full_trace_interval'][1]+value[1]);check(tag+'complete triangle upper',upper==rec['total_TV_upper'] and full[1]<=upper);rows.append(dict(case=index,job=job,arithmetic_TV=value,Gaussian_TV=full,total_TV_upper=upper,tail_mass=rec['bridge']['tail_mass_interval'],compression_trace=rec['bridge']['compression_trace_interval'],full_trace=rec['bridge']['full_trace_interval'],maximum_target_width=max(b-a for a,b in law),failure=word[-1]));print('verified displacement',index,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'comparisons.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(x['passed'] for x in checks),cases=60,Gaussian_constructions=30,source_hashes=hashes,dependencies=deps,numpy_version=np.__version__,scope='Infinite pure Gaussian versus stored finite MPS, exact threshold measurement, independent uniform53-bit word program; fixed displacements only'))
if __name__=='__main__':main()
