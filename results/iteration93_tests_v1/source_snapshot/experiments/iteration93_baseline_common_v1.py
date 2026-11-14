"""Same physical inputs for prior PNR, direct click, pinned PE and Frugal.
Frugal/direct covariance-only adapters are used only on zero-mean inputs.
PE is the unmodified pinned 0.2.2 CPU v2 implementation from iteration92.
"""
from iteration93_quality_common_v1 import *
from iteration89_common_v1 import NativeDirectClickSampler,tree,puretree
from iteration91_common_v1 import optical_cache,optical_sample,uniforms,NativeStableClickSampler
BP=C.POINTER(C.c_uint8)

def old_input(q):
    t=q['t'];alpha=q['alpha'];return dict(v=q['u'],t=-t,x=np.diag(np.r_[t/(1-t),-t/(1+t)]),L=np.sqrt(2)*q['noise'],mean=np.sqrt(2)*np.r_[alpha.real,alpha.imag])

def covariance(q):
    u=q['u'];R=np.block([[u.real,-u.imag],[u.imag,u.real]]);old=old_input(q);X=old['x']+old['L']@old['L'].T;return np.eye(2*len(u))+2*R@X@R.T

class Baseline93:
    def __init__(self,job,q,threads=None):
        self.job=job;self.q=q;self.family=job['family'];self.threads=job['threads'] if threads is None else threads;self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));self.model=None;self.handle=None;self.destroy=None;self.meta={};start=time.perf_counter();m=len(q['u'])
        if self.family=='old':
            self.old=old_input(q);self.model=NativeDirectClickSampler(self.old['v'],self.old['x'],ROOT/'results/iteration89_gates_v1/puretree_probe.dll',runtime_dirs=[Path(shutil.which('g++')).parent]);self.handle,self.destroy=tree(self.model)
        elif self.family=='direct':
            assert not q['alpha'].any();old=old_input(q);self.model=NativeStableClickSampler(old['v'],old['x']+old['L']@old['L'].T,ROOT/'results/iteration91_build_v3/optical.dll',branch_budget=job['budget'],runtime_dirs=[Path(shutil.which('g++')).parent]);self.handle,self.destroy,meta=optical_cache(self.model);self.meta['cache']=meta
        elif self.family=='frugal':
            assert not q['alpha'].any();cov=covariance(q);self.padded=max(4,m);idx=list(range(m))+list(range(self.padded,self.padded+m));self.cov=np.eye(2*self.padded);self.cov[np.ix_(idx,idx)]=cov;self.cov=np.ascontiguousarray(self.cov);self.lib=C.CDLL(str(ROOT/'results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll'));self.fn=self.lib.fc73_sample;self.fn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,DP,BP,DP,C.c_char_p,C.c_int]
        elif self.family=='pe':
            from iteration92_pe_common_v1 import GaussianInput,pe_sample
            self.pe_sample=pe_sample;t=q['t'] if job['case']!=3 else np.array([.85,-.85]);alpha=q['alpha'];phase=np.where(t>0,1j,1);self.U=q['u']*phase;self.state=GaussianInput(np.arctanh(abs(t)),tuple(range(len(t))),m,disp=phase.conj()*alpha);self.loss=.8 if job['case']==3 else None;B,z,logN=self.state.bargmann(self.U);expected=q['u']@np.diag(t)@q['u'].T;beta=q['u']@alpha;assert np.max(abs(B-expected))<2e-14;assert np.max(abs(z-(beta-expected@beta.conj())))<2e-14
            if self.loss is not None:
                R=np.block([[q['u'].real,-q['u'].imag],[q['u'].imag,q['u'].real]]);X=np.diag(np.r_[t/(1-t),-t/(1+t)]);assert np.max(abs(np.eye(2*m)+2*self.loss*R@X@R.T-covariance(q)))<2e-14
            self.meta.update(input_bargmann_error=float(np.max(abs(B-expected))),loss=self.loss,implementation='v2',window=4,device='cpu',threads=1)
        else:raise ValueError(self.family)
        self.meta['setup_seconds']=time.perf_counter()-start

    def run(self,N,offset,experiment,seed=None):
        job=self.job;m=len(self.q['u']);start=time.perf_counter()
        if self.family=='old':
            raw,meta=puretree(self.model,self.handle,self.old,job['depth'],N,experiment,offset=offset,threads=self.threads);raw['failed']=~raw['valid']
        elif self.family=='direct':
            draws=uniforms(experiment,0,offset,N,m);raw,meta=optical_sample(self.model,self.handle,draws,job['budget'],threads=self.threads);raw['failed']=(raw['errors'][:,0]!=0)|(raw['statistics'][:,5]!=m);raw['draws']=draws
        elif self.family=='frugal':
            assert seed is not None;bits=np.zeros((N,self.padded),np.uint8);stats=np.zeros(7);msg=C.create_string_buffer(4096);rc=self.fn(self.padded,job['cutoff'],N,seed,self.threads,ptr(self.cov),bits.ctypes.data_as(BP),ptr(stats),msg,len(msg));raw=dict(clicks=bits[:,:m],padded_clicks=bits,failed=np.full(N,rc!=0),statistics=stats);meta=dict(returncode=rc,error=msg.value.decode(),seed=seed);assert not bits[:,m:].any()
        else:
            assert seed is not None;counts,stats=self.pe_sample(self.state,self.U,N,loss=self.loss,seed=seed,device='cpu',return_stats=True,impl='v2',window=4,max_photons=256);raw=dict(counts=np.asarray(counts),failed=np.asarray(stats.censored,bool),norm_rel_err=np.asarray(stats.norm_rel_err),round_off=np.asarray(stats.round_off),photons=np.asarray(stats.photons),max_lhaf=np.asarray(stats.max_lhaf));meta=dict(seed=seed,max_norm_relative_error=float(max(stats.norm_rel_err,default=0)),max_round_off=float(max(stats.round_off,default=0)))
        meta['wall_seconds_including_rng']=time.perf_counter()-start;meta['failures']=int(raw['failed'].sum());return raw,meta

    def close(self):
        if self.handle:self.destroy(self.handle);self.handle=None
        if self.model:self.model.close();self.model=None
        self.runtime.close()
