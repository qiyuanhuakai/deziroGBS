"""Release checks for support, public entry points, analytic controls and hardware CDFs."""
import argparse,hashlib,json,math,subprocess,sys,tempfile
from fractions import Fraction as F
from pathlib import Path
from time import perf_counter
from threadpoolctl import threadpool_limits
from iteration100_sample_v1 import ROOT,DATA,cached,custom
from iteration100_spatial_v1 import np,H,save,manifest,realized,HIST
from iteration100_projection_v1 import Projector

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    out=ap.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    nart=nsrc=0
    for name in ('spatial','quality','fast','projection','compression','baselines','relabel','residual'):
        folder=DATA/f'iteration100_{name}_v1'
        for path,digest in json.loads((folder/'manifest.json').read_text()).items():
            assert H(folder/path)==digest;nart+=1
        for path,digest in json.loads((folder/'plan.json').read_text())['hashes'].items():
            assert H(ROOT/path)==digest;nsrc+=1
    cli=ROOT/'experiments/iteration100_sample_v1.py'
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),cli,HIST,DATA/'iteration84_photon_cdf_v1/reference.npz')},
        seed=100110000,records=1048576,confidence='three simulated CDFs and one hardware CDF in a joint iid DKW family at alpha=.01'))
    t=perf_counter();model=cached(True);load=perf_counter()-t
    src=model.fast;w=src.width;mask=np.arange(1<<w)
    for i in range(src.leaves):
        for n in range(src.caps[i]+1):
            pm=realized(src.masks[src.leafstarts[i]+n])
            invalid=np.array([m.bit_count()>n or (n>0 and m==0) for m in range(len(pm))])
            assert pm[invalid].sum()==0
    assert max(src.caps)<65535
    hardware=np.load(HIST)['total_click_histogram'];ph=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    root=realized(src.root);tail=float(ph[len(root):].sum())
    root_tv=(float(abs(root-ph[:len(root)]).sum())+tail)/2
    assert root_tv<1e-11
    init=[];tests=[]
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);p=F(1,5);eta=F(3,4);other=F(2,5)
        nn=float(p/(1-p))*np.eye(2);a=float(np.sqrt(float(p))/(1-p))*np.array([[0,1],[1,0]])
        x=np.block([[nn+a,np.zeros((2,2))],[np.zeros((2,2)),nn-a]])
        ports=np.diag(np.sqrt([float(eta),float(other)]));opt=tmp/'loss.npz';np.savez(opt,v=ports,x=x)
        twin,_,_=custom(opt,1);proj=Projector(twin);limit=10;law=proj.photons([0,1],limit)
        expected=np.zeros((limit+1,limit+1),float)
        for k in range(80):
            def loss(e):
                row=np.zeros(limit+1)
                for j in range(k+1):row[min(j,limit)]+=float(math.comb(k,j)*e**j*(1-e)**(k-j))
                return row
            expected+=float((1-p)*p**k)*loss(other)[:,None]*loss(eta)
        tv=float(abs(expected.ravel()-law).sum()/2);assert tv<1e-12
        np.savez(tmp/'pure.npz',v=np.eye(2),x=x)
        pure,_,_=custom(tmp/'pure.npz',1);draw,flags=pure.sample(65536,1,100119000,0,4)
        assert not flags.any() and np.array_equal(draw[:,0],draw[:,1])
        three=np.vstack([ports,[0,0]])
        np.savez(tmp/'three.npz',v=three,x=x);np.savez(tmp/'vac.npz',v=np.eye(3),x=np.zeros((6,6)))
        for j,(name,extra,modes) in enumerate([('actual',[],1152),('tree',['--uncalibrated'],1152),
                ('odd',['--input',str(tmp/'three.npz')],3),('vac',['--input',str(tmp/'vac.npz')],3)]):
            for route in ('click','pnr'):
                output=tmp/f'{name}_{route}.npz';args=[sys.executable,str(cli),str(output),'--route',route,'--shots','128','--workers','2','--seed','100118000',*extra]
                subprocess.run(args,check=True)
                with np.load(output) as f:
                    assert f['records'].shape==(128,modes) and not f['flags'].any()
                    if route=='click':assert f['records'].max()<=1
                    if name=='vac':assert not f['records'].any()
                    if name=='odd':assert not f['records'][:,2].any()
                tests.append([name,route])
    # Include reading and verifying all four prepared/compiled components in the first batch cost.
    for rep in range(5):
        t=perf_counter();cold=cached(True);cold.sample(2048,0,100117000+rep,0,1);init.append((perf_counter()-t)*1e6/2048)
    rows=[]
    for route in (0,1):
        seed=100110000+route;click=np.zeros(1153,np.int64);pnr=np.zeros(2049,np.int64)
        for rep in range(256):
            n=4096;off=4096*rep;z,f,g=model.sample(n,route,seed,off,16)
            assert not f.any()
            click+=np.bincount((z>0).sum(1),minlength=len(click))
            if route:pnr+=np.bincount(z.sum(1).astype(int),minlength=len(pnr))
        ks=float(abs(click.cumsum()/click.sum()-hardware.cumsum()/hardware.sum()).max())
        row=dict(route=route,records=int(click.sum()),hardware_ks=ks,click_mean=float(click@np.arange(len(click))/click.sum()))
        if route:row['photon_ks']=float(abs(pnr.cumsum()[:len(ph)]/pnr.sum()-ph.cumsum()).max())
        rows.append(row);np.savez_compressed(out/f'hist_{route}.npz',click=click,pnr=pnr)
    half=np.sqrt(np.log(800)/2)*(1/np.sqrt(1048576)+1/np.sqrt(hardware.sum()))
    save(out,'validation.json',dict(passed=True,verified_stage_artifacts=nart,verified_stage_inputs=nsrc,
        public_cli=tests,twin_loss_complete_coarse_tv=tv,twin_loss_omitted_upper=float(p**80),root_photon_tv=root_tv,
        root_tail=tail,all_leaf_impossible_mask_weights_zero=True,initial_load_s=load,cached_first_batch_us=init,
        calibrated=rows,joint_hardware_cdf_halfwidth=float(half),hardware_records=int(hardware.sum())))
    manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
