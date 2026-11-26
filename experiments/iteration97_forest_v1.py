"""Sample threshold and number-resolving records across terminal bridges of balanced Fourier cells."""
import argparse,ctypes as ct,math,shutil,subprocess
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
from time import perf_counter
from iteration97_conditionals_v1 import ROOT,Kernel,np,H,save,manifest,I,U,W,B,bind
from iteration96_group_common_v1 import T
from iteration97_bridges_v1 import full_pnr_tv
S=1<<53
def alias_table(weights):
    keys=sorted(weights);values=[weights[k] for k in keys];total=sum(values)
    bits=max(1,(len(keys)-1).bit_length());size=1<<bits;keys.extend([0]*(size-len(keys)))
    scaled=[v*size for v in values]+[0]*(size-len(values));small=[];large=[]
    for i,x in enumerate(scaled):(small if x<total else large).append(i)
    scale=1<<(64-bits);cuts=[scale]*size;alias=list(range(size))
    while small and large:
        i=small.pop();j=large.pop();cuts[i]=scaled[i]*scale//total;alias[i]=j
        scaled[j]+=scaled[i]-total;(small if scaled[j]<total else large).append(j)
    assert all(scaled[i]==total for i in small+large)
    realized=[0]*size
    for i in range(size):realized[i]+=cuts[i];realized[alias[i]]+=scale-cuts[i]
    err=sum(abs(v*2**64-w*total) for v,w in zip(values+[0]*(size-len(values)),realized))
    tv=float(F(err,2*total*2**64))
    return bits,np.array(cuts,np.uint64),np.array(alias,np.uint32),np.array(keys,np.uint32),tv,np.array(realized,dtype=float)/2**64
def transition(n,m,k,r):
    fac=math.factorial;lo=max(0,k-m);hi=min(n,k);c=1-r
    terms=sum((-1)**(n-l)*math.comb(n,l)*math.comb(m,k-l)*c**(l-lo)*r**(hi-l) for l in range(lo,hi+1))
    return F(fac(k)*fac(n+m-k),fac(n)*fac(m))*c**(m-k+2*lo)*r**(n+k-2*hi)*terms**2
def edge_tables(cap,r):
    pnr=np.zeros((cap+1,cap+1,2*cap+1),np.uint64);click=np.zeros((cap+1,cap+1,2),np.uint64);mtv=[0.,0.]
    for n in range(cap+1):
        for m in range(cap+1):
            z=[transition(n,m,k,r) for k in range(n+m+1)]
            assert min(z)>=0 and sum(z)==1
            cdf=F(0);prev=0;err=F(0)
            for k,pr in enumerate(z):
                cdf+=pr;cut=int(cdf*S);pnr[n,m,k]=cut;err+=abs(F(cut-prev,S)-pr);prev=cut
            mtv[1]=max(mtv[1],float(err/2))
            if n+m:
                a=z[0];b=z[-1];cuts=[int(a*S),int((a+b)*S)];click[n,m]=cuts
                aq=F(cuts[0],S);bq=F(cuts[1]-cuts[0],S)
                mtv[0]=max(mtv[0],float((abs(aq-a)+abs(bq-b)+abs(aq+bq-a-b))/2))
    return [click,pnr],mtv
def local_tables(lib,cap,p):
    cell=bind(lib,'forest97_cell',[I,I,T,W]);capacity=sum(math.comb(n+3,3)**2 for n in range(cap+1))
    keys=np.empty(capacity,np.uint32);num=np.empty(capacity,np.uint64);n=cell(cap,capacity,keys,num);assert n>0
    fac=math.factorial;P,D=p.numerator,p.denominator;weights={};coarse=defaultdict(int)
    for code,numer in zip(keys[:n],num[:n]):
        key=int(code);v=[(key>>(4*i))&15 for i in range(8)];t=sum(v[4:]);assert t==sum(v[:4])
        mfac=math.prod(fac(k) for k in v[4:]);w=int(numer)*P**t*(4*D)**(cap-t)*(fac(cap)//mfac)
        weights[key]=w
        clicks=sum(int(v[k]>0)<<i for i,k in enumerate([1,2,4,5,6,7]))
        coarse[v[0]+16*v[3]+256*clicks]+=w
    survival=F(sum(weights.values()),(4*D)**cap*fac(cap))*(1-p)**4
    assert survival==sum(math.comb(n+3,3)*p**n*(1-p)**4 for n in range(cap+1))
    return [alias_table(coarse),alias_table(weights)],survival
def full16(table,bridge):
    _,_,_,keys,_,real=table;cap=bridge.shape[0]-1;L=np.zeros((128,cap+1));R=L.copy()
    for code,mass in zip(keys,real):
        s=int(code);a=s&15;b=(s>>4)&15;bits=s>>8
        L[bits+64*int(a>0),b]+=mass;R[bits+64*int(b>0),a]+=mass
    words=np.arange(65536);lc=sum(((words>>port)&1)<<i for i,port in enumerate([1,2,8,9,10,11,0]))
    rc=sum(((words>>port)&1)<<i for i,port in enumerate([5,6,12,13,14,15,7]))
    observed=(words>>3)&3;law=np.zeros(65536);a=bridge[:,:,0].astype(float)/S;b=bridge[:,:,1].astype(float)/S
    events=[np.zeros_like(a),b-a,a,1-b];events[0][0,0]=1;events[3][0,0]=0
    for pattern,pr in enumerate(events):
        ix=observed==pattern;law[ix]=(L@pr@R.T)[lc[ix],rc[ix]]
    assert abs(law.sum()-1)<2e-12;return law
def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'native/gaussian_forest97_v1.cpp';prev=ROOT/'results/iteration97_bridges_v1'
    files=[Path(__file__),src,ROOT/'native/gaussian_rng_domain84_v1.hpp',prev/'manifest.json']
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},max_cell_pairs=7,
        p='36/625',bridge='reflectivity is binary64 sin(angle)^2 converted to an exact rational',reps=7,seed=97150000,
        guarantee='truncate on cell idler totals, conditional Fourier probabilities and bridge probabilities computed with integers or rational arithmetic'))
    dll=out/'forest.dll';cmd=[shutil.which('g++'),'-shared','-std=c++20','-O3','-fopenmp','-ffp-contract=off',str(src),'-o',str(dll)]
    run=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr));run.check_returncode()
    runtime=Kernel();lib=ct.CDLL(str(dll));draw=bind(lib,'forest97_draw',[I,I,I,I,I,W,T,T,W,I,U,U,I,B])
    p=F(36,625);cap=7;start=perf_counter();tables,surv=local_tables(lib,cap,p);local_prep=perf_counter()-start
    rows=[]
    for j,(cells,theta) in enumerate([(2,.02),(2,.18),(16,.02),(144,.02),(144,.08)]):
        start=perf_counter();r=F.from_float(float(np.sin(theta)**2));edge,errs=edge_tables(cap,r);bridge_prep=perf_counter()-start
        upper=float(1-surv**cells)
        for route in [0,1]:
            bits,cuts,alias,keys,tv,real=tables[route];m=8*cells;dom=97150000+10*j+route
            def sample(n,offset,workers):
                x=np.empty((n,m),np.uint8)
                assert draw(cells,route,bits,cap,cap,cuts,alias,keys,edge[route],n,dom,offset,workers,x)==0
                return x
            row=dict(cells=cells,ports=m,route=route,angle=theta,reflectivity=str(r),local_prep_s=local_prep,
                bridge_prep_s=bridge_prep,alias_states=1<<bits,local_quant_tv=tv,edge_quant_tv=errs[route],
                tail_tv=upper,full_tv_upper=upper+cells*tv+(cells-1)*errs[route])
            if route==0 and cells==2:
                truth=np.load(prev/f'input_{j}.npz')['click'];law=full16(tables[0],edge[0])
                row['full_click_tv']=float(abs(law-truth).sum()/2)
                assert row['full_click_tv']<=row['full_tv_upper']+1e-12
                np.savez_compressed(out/f'law_{j}.npz',law=law)
            x=sample(512,0,1);assert np.array_equal(x[32:],sample(480,32,4))
            times=[];forbidden=0;escape=0;n=2048 if cells==2 else 8192
            for workers in [1,16]:
                stamps=[]
                for rep in range(7):
                    start=perf_counter();x=sample(n,1024+(workers*7+rep)*n,workers);stamps.append((perf_counter()-start)*1e6/n)
                    if route:
                        a=x[:,:m//2].reshape(n,cells,4).sum(2);b=x[:,m//2:].reshape(n,cells,4).sum(2)
                        assert (a.sum(1)==b.sum(1)).all() and b.max()<=cap
                        escape+=int((a!=b).any(1).sum())
                    else:
                        a=x[:,:m//2].reshape(n,cells,4).any(2);b=x[:,m//2:].reshape(n,cells,4).any(2)
                        forbidden+=int((a!=b).any(1).sum())
                times.append(dict(threads=workers,batch=n,us=stamps,median=float(np.median(stamps))))
            event=(escape if route else forbidden)/(14*n)
            if route:assert abs(event-full_pnr_tv(float(p),theta,cells-1))<6*np.sqrt(max(event,1/(14*n))/(14*n))+upper
            row.update(timing=times,observed_event=event,records=14*n)
            rows.append(row);print(m,theta,route,row['full_tv_upper'],times[-1]['median'],flush=True)
    save(out,'summary.json',rows);manifest(out)
if __name__=='__main__':main()
