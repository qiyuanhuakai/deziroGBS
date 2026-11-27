"""Save every scored high-power record and verify the archived timing runs by replay."""
import numpy as np
from threadpoolctl import threadpool_limits
from iteration115_visible_v1 import hardware
from iteration117_quartic import OUT,loadlib,prepare,draw


def run():
    lib=loadlib();t,ss=hardware();batch=32;shots=4096
    for transport in (False,True):
        prefix="transport_" if transport else ""
        models=[prepare(t,s,transport) for s in ss]
        prior=np.load(OUT/(prefix+"records.npz"))
        all_rows={}
        for direct in (0,1):
            scored=[]
            for off in range(0,shots,batch):
                rec=[]
                for j,(model,transform,_) in enumerate(models):
                    domain=[117,510,3,direct,j,off+batch] if transport else [117,510,direct,j,off+batch]
                    rng=np.random.default_rng(np.random.SeedSequence(domain))
                    alpha=rng.normal(size=(batch,len(transform)))@transform
                    result,diag,_=draw(lib,model,alpha,direct,j+6*transport,off,8)
                    assert diag.max()<1e-6
                    rec.append(result)
                rows=rec[0]|rec[1] if direct else rec[0].astype(np.uint16)+rec[1]
                scored.append(rows)
            labels=np.concatenate(scored)
            banks=(labels>0).reshape(shots,144,8).sum(2)
            np.testing.assert_array_equal(labels[:16],prior[f"sample_{direct}"])
            np.testing.assert_array_equal(np.bincount(banks.sum(1),minlength=1153),prior[f"clicks_{direct}"])
            np.testing.assert_array_equal(np.bincount(9*banks[:,34]+banks[:,35],minlength=81),prior[f"panel_{direct}"])
            if not direct:np.testing.assert_array_equal(np.bincount(labels.sum(1).astype(int),minlength=1153),prior["photons"])
            all_rows["threshold" if direct else "pnr"]=labels
        np.savez_compressed(OUT/(prefix+"scored_4096.npz"),**all_rows)
        print(prefix or "untransported",shots,"verified per readout",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=3):run()
