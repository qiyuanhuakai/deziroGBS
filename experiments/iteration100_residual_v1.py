"""Identify the strongest missing spatial covariance and directly verify the resulting count and click events."""
import argparse,ctypes as ct,json
from pathlib import Path
from scipy.stats import beta
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,np,Tree,Native,I,D,bind,H,save,manifest,realify,split_joint,ORDER
from iteration100_projection_v1 import Projector,physical

def covariance(proj):
    model=proj.model;L=model.leaves
    cross=np.zeros((L,L));columns={};response={}
    for i in range(len(model.caps)):
        q=proj.totals[i];n=np.arange(len(q))
        if i<L:
            columns[i]=[i];response[i]=n[:,None];cross[i,i]=q@(n*n)
        else:
            a,b=model.children[i];u=response.pop(a);v=response.pop(b);ia=columns.pop(a);ib=columns.pop(b)
            block=u.T@proj.joint[i]@v
            cross[np.ix_(ia,ib)]=block;cross[np.ix_(ib,ia)]=block.T
            columns[i]=ia+ib;response[i]=np.c_[proj.conditional[i][0]@u,proj.conditional[i][1]@v]
    means=np.array([proj.totals[g]@np.arange(len(proj.totals[g])) for g in range(L)])
    return cross-means[:,None]*means

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    out=ap.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';base=data/'iteration100_spatial_v1';prior=data/'iteration100_relabel_v1/summary.json'
    qdll=data/'iteration96_quad_v1/oracle.dll';files=[Path(__file__),base/'prepared.npz',ORDER,prior,qdll]
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},
        selection='top three disjoint pairs by absolute residual covariance divided by optical standard deviations, among spatial modes 16..143',
        confidence='two one-sided Clopper-Pearson limits each at alpha=.005 on previously generated independent correction-spill flags'))
    lib=Native(base/'spatial.dll');oracle=bind(ct.CDLL(str(qdll)),'oracle96',[I,I,D,D,D,D])
    with np.load(base/'prepared.npz') as f:model=Tree(lib,dict(f))
    proj=Projector(model);found=covariance(proj)
    z=np.load(ORDER);v=z['v'][np.argsort(z['order'])];L,w=model.ports.shape
    spatial=np.array([realify(v[i*w:(i+1)*w]) for i in range(L)]);grams=spatial.transpose(0,2,1)@spatial
    target=sum(.5*(x@grams@x).reshape(L,-1)@grams.reshape(L,-1).T for x in z['x'])
    target+=np.diag(sum(np.einsum('ij,gji->g',x,grams) for x in z['x'])/2)
    gaps=(target-found)/np.sqrt(target.diagonal()[:,None]*target.diagonal())
    objective=abs(gaps);np.fill_diagonal(objective,0);objective[:16]=0;objective[:,:16]=0
    prev=json.loads(prior.read_text());spill=[]
    for row in prev['rows']:
        k=sum(row['flag_counts'][i] for i in range(8) if i&4)
        spill.append(float(beta.ppf(.995,k+1,row['records']-k)))
    rows=[]
    for j in range(3):
        a,b=np.unravel_index(objective.argmax(),objective.shape)
        objective[[a,b]]=0;objective[:,[a,b]]=0
        ports=np.r_[np.arange(w)+w*a,np.arange(w)+w*b];q=physical(z,ports);truth=np.empty(65536)
        assert oracle(16,2,q,np.zeros((2,32)),truth,np.empty(2))==0
        predicted=proj.clicks(ports);tv=float(abs(predicted-truth).sum()/2)
        ca,cb=map(int,model.caps[[a,b]])
        ref=split_joint(lib,grams[a],grams[b],z['x'],ca,cb)
        def numbers(i):return np.eye(model.caps[i]+1),[i]
        law,order=proj.response({a,b},numbers)
        law=law.reshape((cb+1,ca+1)).T if order==[a,b] else law.reshape((ca+1,cb+1))
        ct_tv=float(abs(law-ref).sum()/2)
        # Cross-check the bivariate inversion by direct determinants at real generating variables.
        na=np.arange(ca+1);nb=np.arange(cb+1);error=0.
        for t,u in [(0.,0.),(.2,.8),(.8,.2),(.99,.95),(1.,1.)]:
            logg=-sum(np.linalg.slogdet(np.eye(len(x))+x@((1-t)*grams[a]+(1-u)*grams[b]))[1] for x in z['x'])/2
            error=max(error,abs(float(t**na@ref@u**nb)-np.exp(logg)))
        assert error<2e-12
        rows.append(dict(spatial=[int(a),int(b)],optical_cov=float(target[a,b]),tree_cov=float(found[a,b]),normalized_residual=float(gaps[a,b]),
            full16_click_tv=tv,spatial_two_counts_tv=ct_tv,pgf_max_error=error))
        np.savez_compressed(out/f'pair_{j}.npz',click=predicted,truth=truth,counts=law,count_reference=ref)
    click=max(r['full16_click_tv'] for r in rows);pnr=max([click]+[r['spatial_two_counts_tv'] for r in rows]);tol=3e-9
    q=prev['outside_region_change_upper'];s=max(spill)
    save(out,'summary.json',dict(rows=rows,tree_1152_tv_lower=[click-tol,pnr-tol],spill_joint99_upper=spill,
        calibrated_1152_tv_lower_deterministic=[max(0,click-q-tol),max(0,pnr-q-tol)],
        calibrated_1152_tv_lower_joint99=[max(0,click-s-tol),max(0,pnr-s-tol)]))
    np.savez_compressed(out/'covariance.npz',optical=target,model=found);manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
