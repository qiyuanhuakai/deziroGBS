"""Postselect unused spatial splitter modes in vacuum and quantify the probability cost."""
import argparse,ctypes as ct,json
from pathlib import Path
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,np,Native,I,D,bind,H,save,manifest,realify,ORDER

def occupation(x):
    w=len(x)//2
    return (x[:w,:w]+x[w:,w:]+1j*(x[w:,:w]-x[:w,w:]))/2

def condition(spectra,discard):
    d=realify(discard);eye=np.eye(len(d));lp=0.;out=[]
    ev,vec=np.linalg.eigh((d+d.T)/2)
    reduced=np.sqrt(np.maximum(ev,0))[:,None]*vec.T
    check=0.
    for x in spectra:
        a=eye+x@d
        sign,log=np.linalg.slogdet(a);assert sign==1
        alt=np.linalg.eigvalsh(reduced@x@reduced.T)
        check=max(check,float(abs(log-np.log1p(alt).sum())))
        lp-=log/2
        y=np.linalg.solve(a,x);assert abs(y-y.T).max()<2e-12
        out.append((y+y.T)/2)
    assert lp<1e-12 and check<2e-11
    return np.array(out),lp,check

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);out=ap.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';base=data/'iteration100_spatial_v1';proj=data/'iteration100_projection_v1'
    qdll=data/'iteration96_quad_v1/oracle.dll'
    files=[Path(__file__),ORDER,base/'prepared.npz',qdll,proj/'summary.json']
    save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},ranks=[1,2,4,6,8],
         optical_bound='trace distance to the normalized joint projection onto discarded vacuum is at most sqrt(1-p_vac), including mixed inputs by purification'))
    lib=Native(base/'spatial.dll');oracle=bind(ct.CDLL(str(qdll)),'oracle96',[I,I,D,D,D,D])
    z=np.load(ORDER);v=z['v'][np.argsort(z['order'])].reshape(144,8,50);x=z['x']
    groups=json.loads((proj/'summary.json').read_text())['panels']
    eig,u=np.linalg.eigh(v@sum(map(occupation,x))@v.conj().transpose(0,2,1))
    u=u[:,:,::-1];rows=[]
    for rank in (1,2,4,6,8):
        kept=u[:,:,:rank];small=kept.conj().transpose(0,2,1)@v
        residual=v-kept@small
        r=residual.reshape(-1,50);discard=r.conj().T@r
        y,lp,diff=condition(x,discard)
        leak=float(np.trace(sum(map(occupation,x))@discard).real)
        tv=float(np.sqrt(max(0,-np.expm1(lp))))
        # Use the identity transformation exactly when all eight branches are retained.
        if rank==8:
            assert abs(lp)<1e-12 and np.linalg.norm(r)<1e-11
            y=x;lp=0.;leak=0.;tv=0.
        output=(kept@small).reshape(1152,50);panels=[]
        for j,row in enumerate(groups):
            ports=row['ports'];t=realify(output[ports]);q=np.array([np.eye(len(t))+t@s@t.T for s in y])
            law=np.empty(1<<len(ports));assert oracle(len(ports),2,q,np.zeros((2,len(t))),law,np.empty(2))==0
            ref=np.load(proj/f'panel_{j}.npz')['truth']
            gap=float(abs(ref-law).sum()/2);assert gap<=tv+1e-12
            panels.append(gap)
        if rank==1:
            np.savez_compressed(out/'compressed144.npz',kept_transfer=small[:,0,:],source_x=y,splitter=kept[:,:,0],log_success=lp)
        rows.append(dict(rank=rank,spatial_modes=144*rank,expected_discarded=leak,joint_vacuum_probability=float(np.exp(lp)),
                         complete_measurement_tv_upper=tv,determinant_log_disagreement=diff,panel_tv=panels))
    # Bound the PNR allocation conditional on a local bitmask. A single
    # surviving spatial mode splits multinomially even with two unresolved spectra.
    ports=np.load(base/'prepared.npz')['ports']
    local=[]
    for i,g in enumerate(v):
        phi=np.sqrt(ports[i])*np.exp(1j*np.angle(u[i,:,0]))
        res=g-phi[:,None]*(phi.conj()@g)
        _,lp,_=condition(x,res.conj().T@res)
        local.append(2*float(np.sqrt(max(0,-np.expm1(lp)))))
    save(out,'summary.json',dict(rows=rows,local_pnr_model_tv_upper=local,max_local_pnr_bound=max(local),
            scope='local bound assumes exact local total and exact mask conditional law; include numerical tree residual when comparing native output'))
    manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
