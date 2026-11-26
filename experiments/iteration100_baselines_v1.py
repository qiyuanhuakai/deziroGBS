"""Equal-width Frugal click comparisons and purified real-input PNR comparisons."""
import argparse,ctypes as ct,itertools,json,math,subprocess
from decimal import Decimal,localcontext
from pathlib import Path
from time import perf_counter
import mpmath as mp
from scipy.linalg import null_space
from scipy.stats import beta
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,np,Native,Tree,prepare,realify,ORDER,H,save,manifest,bind,I,D,B
from iteration100_fast_v1 import Fast,make
from iteration100_projection_v1 import Projector,physical
from iteration92_pe_common_v1 import GaussianInput,pe_sample,UPSTREAM

def independent_or(laws):
    with localcontext() as c:
        c.prec=64;m=int(math.log2(len(laws[0])));cdf=np.full(len(laws[0]),Decimal(1),dtype=object)
        for v in laws:
            x=np.array([Decimal.from_float(float(t)) for t in v],object)
            for j in range(m):
                a=x.reshape(-1,2,1<<j);a[:,1]+=a[:,0]
            cdf*=x
        for j in range(m):
            a=cdf.reshape(-1,2,1<<j);a[:,1]-=a[:,0]
    return cdf.astype(float)

def frugal(lib,q,cutoff,seed):
    tree=bind(lib,'fc96_full16',[I,D,D,D,D,D,ct.c_char_p,I])
    sampler=bind(lib,'fc96_sample16',[I,I,ct.c_uint,D,B,D,ct.c_char_p,I])
    msg=ct.create_string_buffer(2048);laws=[];cov=2*q-np.eye(32)
    for v in cov:
        th=np.empty(65535);prev=th.copy();prop=th.copy();stats=np.zeros(3)
        assert tree(cutoff,v,th,prev,prop,stats,msg,len(msg))==0 and stats[2]==64
        prob=np.ones(1,np.longdouble)
        for j in range(16):
            zero=np.nan_to_num(th[len(prob)-1:2*len(prob)-1],nan=1).clip(0,1).astype(np.longdouble)
            prob=np.r_[prob*zero,prob*(1-zero)]
        laws.append(prob)
    times=[]
    for rep in range(6):
        start=perf_counter();samples=[]
        for j,v in enumerate(cov):
            x=np.empty((2048,16),np.uint8);s=np.empty(7)
            assert sampler(cutoff,2048,seed+2*rep+j,v,x,s,msg,len(msg))==0 and s[5]==64
            samples.append(x)
        x=np.bitwise_or.reduce(samples)
        if rep:times.append((perf_counter()-start)*1e6/2048)
    return independent_or(laws),times

def purification(q):
    v=2*q-np.eye(len(q));m=len(v)//2;eye=np.eye(m)
    omega=np.block([[0*eye,eye],[-eye,0*eye]])
    vals,o=np.linalg.eigh(v);root=(o*np.sqrt(vals))@o.T
    a=root@omega@root;n2,u=np.linalg.eigh(-a@a)
    assert n2.min()>1-2e-12
    mirror=np.diag(np.r_[np.ones(m),-np.ones(m)])
    cross=root@(u*np.sqrt(np.maximum(0,1-1/n2)))@u.T@root@mirror
    both=np.block([[v,cross],[cross.T,mirror@v@mirror]])
    order=np.r_[0:m,2*m:3*m,m:2*m,3*m:4*m];both=both[np.ix_(order,order)]
    e,f=np.linalg.eigh(both);hot=e>1+1e-10
    modes=-1j*(f[:2*m,hot]+1j*f[2*m:,hot]);r=np.log(e[hot])/2
    if len(r)<2*m:modes=np.column_stack([modes,null_space(modes.conj().T)])
    assert abs(modes.conj().T@modes-np.eye(2*m)).max()<1e-10
    t=np.r_[r,np.zeros(2*m-len(r))];z=realify(modes)
    err=float(abs((z*np.exp(np.r_[-2*t,2*t]))@z.T-both).max());assert err<1e-9
    return GaussianInput(r,tuple(range(len(r))),2*m),modes,err

def pair_reference(q,lim):
    # Coefficients of det(I-ZK)^(-1/2) from principal minors and its Euler derivative.
    with mp.workdps(65):
        def coefficients(mats,dim):
            poly={(0,)*dim:mp.mpf(1)};vac=mp.mpf(1)
            for raw in mats:
                m=mp.matrix(raw.tolist());vac/=mp.sqrt(mp.det(m));kernel=mp.eye(2*dim)-m**-1
                part={}
                for mask in range(1<<(2*dim)):
                    ids=[i for i in range(2*dim) if mask>>i&1]
                    key=tuple(sum(i%dim==j for i in ids) for j in range(dim))
                    det=mp.det(mp.matrix([[kernel[i,j] for j in ids] for i in ids])) if ids else 1
                    part[key]=part.get(key,0)+(-1)**len(ids)*det
                nxt={}
                for a,x in poly.items():
                    for b,y in part.items():
                        key=tuple(i+j for i,j in zip(a,b));nxt[key]=nxt.get(key,0)+x*y
                poly=nxt
            p={(0,)*dim:vac}
            for n in itertools.product(range(lim),repeat=dim):
                total=sum(n)
                if not total:continue
                p[n]=-sum((2*total-sum(a))*v*p[tuple(i-j for i,j in zip(n,a))] for a,v in poly.items()
                    if sum(a) and all(i>=j for i,j in zip(n,a)))/(2*total)
            return np.array([float(p[n]) for n in itertools.product(range(lim),repeat=dim)]).reshape((lim,)*dim)
        mat=np.zeros((lim+1,lim+1));mat[:lim,:lim]=coefficients(q,2).T
        for j in (0,1):
            ids=[j,j+2];pr=coefficients(q[:,ids][:,:,ids],1)
            if j==0:mat[-1,:lim]=pr-mat[:lim,:lim].sum(0)
            else:mat[:lim,-1]=pr-mat[:lim,:lim].sum(1)
        mat[-1,-1]=1-mat.sum()
        assert mat.min()>-2e-14
        return mat.ravel()

def main():
    arg=argparse.ArgumentParser();arg.add_argument('output',type=Path);out=arg.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';base=data/'iteration100_spatial_v1';fd=data/'iteration100_fast_v1/fast.dll'
    panels=data/'iteration100_projection_v1';frd=data/'iteration96_frugal_build_v1/frugal.dll';z=np.load(ORDER)
    files=[Path(__file__),ORDER,base/'spatial.dll',base/'prepared.npz',fd,frd,*(panels/f'panel_{j}.npz' for j in (0,1,4))]
    save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},seed=100070000,
        pe_revision=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
        comparison='same observed width, spectra and physical inputs; 2048 records/call, one CPU thread; PE purifies each actual mixed two-port state'))
    lib=Native(base/'spatial.dll');flib=ct.CDLL(str(frd))
    with np.load(base/'prepared.npz') as f:full=Projector(Tree(lib,dict(f)))
    original=z['v'][np.argsort(z['order'])];clickrows=[];pnrrows=[]
    def reduced(ports,width):
        start=perf_counter();tables,diag=prepare(lib,original[ports],z['x'],width)
        extra,transfer=make(Tree(lib,tables));model=Fast(fd,tables,extra)
        return model,Projector(model),perf_counter()-start
    for j in (0,1,4):
        ref=np.load(panels/f'panel_{j}.npz');ports=ref['ports'];truth=ref['truth'];q=ref['q']
        model,projection,prep=reduced(ports,8 if j==4 else 1)
        law=projection.clicks(np.arange(16));us=[]
        for rep in range(7):
            t=perf_counter();x,e=model.sample(2048,0,100070000+j,2048*rep,1);us.append((perf_counter()-t)*1e6/2048);assert not e.any()
        row=dict(panel=j,ports=ports.tolist(),tree_tv=float(abs(law-truth).sum()/2),tree_us=us,median=float(np.median(us)),prep_s=prep,frugal=[])
        for k in (2,4):
            other,times=frugal(flib,q,k,100071000+200*j+30*k)
            row['frugal'].append(dict(K=k,complete_tv=float(abs(other-truth).sum()/2),us=times,median=float(np.median(times))))
            np.savez_compressed(out/f'frugal_{j}_{k}.npz',law=other)
        clickrows.append(row);print('panel',j,row['tree_tv'],flush=True)
    for j,ports in enumerate(([1001,1005],[1001,706])):
        q=physical(z,ports);lim=12;truth=pair_reference(q,lim)
        model,pp,prep=reduced(ports,1);law=pp.photons([0,1],lim);full_law=full.photons(ports,lim)
        upper_tail=float(truth.reshape(lim+1,lim+1)[-1].sum()+truth.reshape(lim+1,lim+1)[:-1,-1].sum())
        prepared=[purification(c) for c in q]
        def pe(n,seed):
            total=np.zeros((n,2),np.uint32)
            for h,(state,modes,_) in enumerate(prepared):
                x,st=pe_sample(state,modes,n,seed=seed+h,device='cpu',return_stats=True,impl='v2',window=4,max_photons=64)
                assert not np.any(st.censored)
                total+=np.asarray(x[:,:2],dtype=np.uint32)
            return total
        pe(32,100080000+j*1000);tree_us=[];pe_us=[]
        for rep in range(5):
            start=perf_counter();x,e=model.sample(2048,1,100080100+j,rep*2048,1);tree_us.append((perf_counter()-start)*1e6/2048)
            start=perf_counter();pe(2048,100081000+j*1000+2*rep);pe_us.append((perf_counter()-start)*1e6/2048)
        n=32768;xs=np.concatenate([pe(2048,100084000+j*1000+2*rep) for rep in range(n//2048)])
        ours,e=model.sample(n,1,100087000+j,0,1);assert not e.any()
        def empirical(a):
            codes=np.minimum(a,lim)@np.array([1,lim+1])
            return float(abs(np.bincount(codes,minlength=len(truth))/len(a)-truth).sum()/2)
        tv=float(abs(law-truth).sum()/2);ftv=float(abs(full_law-truth).sum()/2)
        row=dict(ports=ports,threshold=lim,reference_tail=upper_tail,reduced_coarse_tv=tv,full_output_pair_tv=[ftv,ftv+upper_tail],
            reduced_full_tv_upper=tv+upper_tail,tree_us=tree_us,pe_us=pe_us,prep_s=prep,records=n,
            tree_empirical=empirical(ours),pe_empirical=empirical(xs),pe_covariance_max_error=max(x[2] for x in prepared))
        np.savez_compressed(out/f'pnr_{j}.npz',truth=truth,full=full_law,reduced=law,pe=xs,ours=ours)
        pnrrows.append(row);print('pair',j,tv,ftv,flush=True)
    save(out,'summary.json',dict(click=clickrows,pnr=pnrrows));manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
