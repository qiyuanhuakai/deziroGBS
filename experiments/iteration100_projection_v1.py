"""Contract the implemented integer-CDF tree to complete optical marginal distributions."""
import argparse,ctypes as ct,json,math
from pathlib import Path
from time import perf_counter
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,np,Native,Tree,H,save,manifest,realized,realify,S,I,D,bind,ORDER

class Projector:
    def __init__(self,model):
        self.model=model;self.L=model.leaves
        self.conditional={};self.joint={};self.totals={len(model.caps)-1:realized(model.root)}
        self.overflow=0.;self.mask=[]
        for i in range(len(model.caps)-1,self.L-1,-1):
            a,b=model.children[i];ca,cb=map(int,model.caps[[a,b]])
            true=self.totals[i];weights=np.zeros((ca+1,cb+1))
            ta=np.zeros((len(true),ca+1));tb=np.zeros((len(true),cb+1))
            for n,v in enumerate(true):
                p=model.starts[i]+n*(n+1)//2;q=realized(model.splits[p:p+n+1]);k=np.arange(n+1)
                inside=(k<=ca)&(n-k<=cb);k=k[inside]
                self.overflow+=v*q[~inside].sum()
                weights[k,n-k]=v*q[k];ta[n,k]=q[k];tb[n,n-k]=q[k]
            self.joint[i]=weights;self.conditional[i]=(ta,tb)
            self.totals[a]=weights.sum(1);self.totals[b]=weights.sum(0)
        for g in range(self.L):
            start=model.leafstarts[g]
            self.mask.append(realized(model.masks[start:start+model.caps[g]+1]))
        assert self.overflow<1e-11
    def response(self,group,feature):
        # feature maps a local count and mask into probabilities of requested outcomes.
        # Only explicit leaves in this dict enter the contraction.
        model=self.model;selected=set(group)
        def walk(i,unconditional=False):
            if i<self.L:
                if i not in selected:return None,[]
                a,ids=feature(i)
                return (self.totals[i]@a if unconditional else a),ids
            left,right=model.children[i];a,la=walk(left);b,lb=walk(right)
            if a is None and b is None:return None,[]
            if a is None or b is None:
                mat,ids=(a,la) if b is None else (b,lb)
                side=int(a is None)
                if unconditional:return self.totals[[left,right][side]]@mat,ids
                return self.conditional[i][side]@mat,ids
            if unconditional:return (b.T@self.joint[i].T@a).ravel(),la+lb
            ca,cb=a.shape[0]-1,b.shape[0]-1
            cond=self.conditional[i][0];ans=np.zeros((len(cond),a.shape[1]*b.shape[1]))
            for n in range(len(cond)):
                lo=max(0,n-cb);hi=min(n,ca)
                if hi>=lo:
                    ix=np.arange(lo,hi+1);ans[n]=((b[n-ix]*cond[n,ix,None]).T@a[ix]).ravel()
            return ans,la+lb
        # Remove unary ancestry before constructing any large conditional word arrays.
        nodemap=[{i} for i in range(self.L)]
        for a,b in model.children[self.L:]:nodemap.append(nodemap[a]|nodemap[b])
        root=next(i for i,s in enumerate(nodemap) if selected<=s)
        return walk(root,True)
    def clicks(self,ports):
        ports=list(map(int,ports));w=self.model.width;groups={p//w for p in ports}
        def leaf(i):
            ids=[p for p in ports if p//w==i];words=np.arange(1<<w)
            labels=sum(((words>>(p%w))&1)<<j for j,p in enumerate(ids));q=self.mask[i]
            ans=np.column_stack([q[:,labels==j].sum(1) for j in range(1<<len(ids))])
            return ans,ids
        q,order=self.response(groups,leaf)
        labs=np.arange(len(q));idx=sum(((labs>>j)&1)<<ports.index(port) for j,port in enumerate(order))
        result=np.empty_like(q);result[idx]=q
        assert abs(result.sum()-1)<1e-10 and result.min()>=0
        return result
    def photons(self,ports,limit=8):
        # Joint probabilities for one or two observed ports. Last coordinate bin
        # includes all counts >= limit, calculated by summing the residual mass.
        w=self.model.width;ports=list(map(int,ports));base=limit+1
        assert len(ports) in (1,2)
        def leaf(i):
            ids=[p for p in ports if p//w==i];local=[p%w for p in ids]
            mask_law=self.mask[i];nmax=len(mask_law)-1
            prior=self.model.prior[self.model.leafstarts[i]:self.model.leafstarts[i]+nmax+1]
            weights=self.model.ports[i];allmask=np.arange(1<<w)
            ratio=np.divide(mask_law,prior,out=np.zeros_like(prior),where=prior>0)
            def fixed(n,which,k):
                if sum(k)>n:return 0.
                allowed=np.ones(1<<w,bool);remove=0;factor=1.;remain=n
                for j,v in zip(which,k):
                    allowed&=((allmask>>j)&1)==(v>0);remove|=1<<j
                    factor*=math.comb(remain,v)*weights[j]**v;remain-=v
                return factor*(ratio[n,allowed]@prior[remain,allmask[allowed]&~remove])
            rows=[]
            for n in range(nmax+1):
                left=np.array([fixed(n,[local[0]],[k]) for k in range(limit)])
                if len(ids)==1:p=np.r_[left,1-left.sum()]
                else:
                    right=np.array([fixed(n,[local[1]],[k]) for k in range(limit)])
                    mat=np.zeros((base,base))
                    mat[:limit,:limit]=[[fixed(n,local,[a,b]) for a in range(limit)] for b in range(limit)]
                    mat[-1,:limit]=left-mat[:limit,:limit].sum(0);mat[:limit,-1]=right-mat[:limit,:limit].sum(1)
                    mat[-1,-1]=1-mat.sum();p=mat.ravel()
                rows.append(np.maximum(p,0))
            return np.array(rows),ids
        q,ids=self.response({p//w for p in ports},leaf)
        if ids!=ports:q=q.reshape(base,base).T.ravel()
        assert abs(q.sum()-1)<1e-12
        return q

def physical(z,ids):
    v=z['v'][np.argsort(z['order'])][ids];a=realify(v)
    return np.array([np.eye(len(a))+a@x@a.T for x in z['x']])

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output.resolve()
    out.mkdir(parents=True,exist_ok=False);data=ROOT/'results';base=data/'iteration100_spatial_v1'
    prior=data/'iteration99_release_v1';qd=data/'iteration96_quad_v1/oracle.dll';z=np.load(ORDER)
    controls=json.loads((prior/'validation.json').read_text())['heldout']
    groups=[np.arange(16)*8]+[np.array(x['ports']) for x in controls]
    lib=Native(base/'spatial.dll')
    oracle=bind(ct.CDLL(str(qd)),'oracle96',[I,I,D,D,D,D])
    with np.load(base/'prepared.npz') as cache:model=Tree(lib,dict(cache))
    proj=Projector(model)
    # Compare a complete sibling of the brightest spatial block, as well as the pre-existing panels.
    bright=int(groups[1][0]//8)
    parent=next((i for i in range(model.leaves,len(model.caps)) if bright in model.children[i] and max(model.children[i])<model.leaves),None)
    assert parent is not None
    groups.append(np.concatenate([np.arange(8)+g*8 for g in model.children[parent]]))
    files=[Path(__file__),ORDER,qd,base/'prepared.npz',prior/'validation.json',*(prior/f'heldout_{i}.npz' for i in range(3))]
    save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},groups=[g.tolist() for g in groups]))
    rows=[]
    for i,ports in enumerate(groups):
        start=perf_counter();q=proj.clicks(ports);seconds=perf_counter()-start
        cov=physical(z,ports);truth=np.empty(len(q))
        assert oracle(len(ports),2,cov,np.zeros((2,2*len(ports))),truth,np.empty(2))==0
        tv=float(abs(q-truth).sum()/2);row=dict(ports=ports.tolist(),full_click_tv=tv,contraction_s=seconds)
        if 1<=i<=3:
            ref=np.load(prior/f'heldout_{i-1}.npz');assert abs(ref['truth']-truth).sum()<1e-13
            event=ref['event'];row['frozen_exact_gap']=float((q-truth)[event].sum())
        np.savez_compressed(out/f'panel_{i}.npz',law=q,truth=truth,q=cov,ports=ports)
        rows.append(row);print(i,tv,seconds,flush=True)
    local=[]
    from iteration100_spatial_v1 import leaf_table
    for g in range(model.leaves):
        ids=np.arange(8)+8*g;cov=physical(z,ids);truth=np.empty(256)
        assert oracle(8,2,cov,np.zeros((2,16)),truth,np.empty(2))==0
        predicted=proj.totals[g]@proj.mask[g]
        local.append(float(abs(truth-predicted).sum()/2))
    save(out,'summary.json',dict(panels=rows,leaf_full_click_tv=local,max_8bit_tv=max(local),lost_probability=proj.overflow))
    manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=4,user_api='blas'):main()
