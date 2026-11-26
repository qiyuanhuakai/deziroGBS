"""Frozen physical-port tests against the previous iteration, optical PMF and hardware."""
import argparse,json
from pathlib import Path
from time import perf_counter
from scipy.stats import beta
from iteration100_spatial_v1 import ROOT,np,Native,Tree,HIST,H,save,manifest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    out=ap.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';base=data/'iteration100_spatial_v1';last=data/'iteration99_release_v1'
    rec=json.loads((last/'validation.json').read_text());hfiles=[last/f'heldout_{i}.npz' for i in range(3)]
    ports=[np.array(d['ports']) for d in rec['heldout']]
    controls=[np.load(p) for p in hfiles];old=[z['event'] for z in controls]
    photon=data/'iteration84_photon_cdf_v1/reference.npz';trueN=np.load(photon)['photon_pmf']
    hardware=np.load(HIST)['total_click_histogram']
    deps=[Path(__file__),base/'spatial.dll',base/'prepared.npz',last/'validation.json',*hfiles,HIST,photon]
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in deps},validation=1048576,
        seed=100010000,scope='three frozen events plus three fresh pilot-selected one-sided events per route; simultaneous two-sided exact binomial intervals for twelve probabilities, alpha=.01',
        hardware='two simulated CDFs and one hardware CDF, DKW alpha=.01/3 per CDF; photon CDF reported descriptively'))
    lib=Native(base/'spatial.dll');start=perf_counter()
    with np.load(base/'prepared.npz') as z:model=Tree(lib,dict(z))
    load=perf_counter()-start;rows=[]
    for route in (0,1):
        dom=100010000+route;n=4096
        x,err=model.sample(512,route,dom,0,1);y,f=model.sample(480,route,dom,32,4)
        assert np.array_equal(x[32:],y) and np.array_equal(err[32:],f)
        pilot=[np.zeros(len(v),np.int64) for v in old];valid=[x.copy() for x in pilot]
        click=np.zeros(1153,np.int64);pnr=np.zeros(2049,np.int64);fails=0
        for target,reps,off in [(pilot,32,4096),(valid,256,2**20)]:
            for rep in range(reps):
                x,f=model.sample(n,route,dom,off+rep*n,16);fails+=int((f!=0).sum())
                b=x>0
                for hist,group in zip(target,ports):
                    hist+=np.bincount(b[:,group]@(1<<np.arange(len(group))),minlength=len(hist))
                if target is valid:
                    click+=np.bincount(b.sum(1),minlength=len(click))
                    if route:pnr+=np.bincount(x.sum(1).astype(int),minlength=len(pnr))
                if rep==0 and route:
                    direct,df=model.sample(n,0,dom,off,16);assert np.array_equal(direct,b) and np.array_equal(df,f)
        assert not fails
        panel=[]
        for i,(h,v,z,m) in enumerate(zip(pilot,valid,controls,old)):
            entry={}
            for label,event in [('frozen',m),('fresh',h/h.sum()>z['truth'])]:
                k=int(v[event].sum());size=int(v.sum());mass=float(z['truth'][event].sum())
                interval=[float(beta.ppf(.01/24,k,size-k+1)) if k else 0.,float(beta.ppf(1-.01/24,k+1,size-k)) if k<size else 1.]
                entry[label]=dict(count=k,optical_probability=mass,observed_gap=k/size-mass,gap_interval=[t-mass for t in interval])
            panel.append(entry);np.savez_compressed(out/f'panel_{route}_{i}.npz',pilot=h,validation=v)
        times=[]
        for workers in (1,16):
            us=[]
            for rep in range(7):
                start=perf_counter();model.sample(8192,route,dom,2**24+(workers*7+rep)*8192,workers)
                us.append((perf_counter()-start)*1e6/8192)
            times.append(dict(workers=workers,us=us,median=float(np.median(us))))
        ks=float(abs(click.cumsum()/click.sum()-hardware.cumsum()/hardware.sum()).max())
        row=dict(route=route,records=int(click.sum()),hardware_ks=ks,timing=times,panels=panel,flags=fails)
        if route:row['optical_photon_ks']=float(abs(pnr.cumsum()[:len(trueN)]/pnr.sum()-trueN.cumsum()).max())
        rows.append(row);np.savez_compressed(out/f'hist_{route}.npz',click=click,pnr=pnr)
        print(route,ks,times[-1]['median'],[p['frozen']['observed_gap'] for p in panel],flush=True)
    half=np.sqrt(np.log(600)/2)*(1/np.sqrt(hardware.sum())+1/np.sqrt(1048576))
    save(out,'summary.json',dict(rows=rows,cache_load_s=load,hardware_n=int(hardware.sum()),hardware_joint_halfwidth=float(half)))
    manifest(out)
if __name__=='__main__':main()
