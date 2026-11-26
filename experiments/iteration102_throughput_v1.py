"""Select thread/batch settings on pilot timings, then independently retime the same sampler."""
import argparse
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration102_sample_v1 import Sampler,RUN,save,manifest

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    cold=[]
    for i in range(3):
        start=perf_counter()
        m=Sampler(RUN/'forest.dll',depth=4)
        m.sample(2048,False,True,16,102120000+i)
        cold.append((perf_counter()-start)*1e6/2048)
    physical=min(64,os.cpu_count() or 1)
    settings=[(w,n) for w in sorted({min(physical,w) for w in (8,12,16,20,24)}) for n in (2048,8192,32768)]
    rng=np.random.default_rng(102120010)
    rows=[]
    for route in (False,True):
        times={s:[] for s in settings}
        for rep in range(4):
            for index in rng.permutation(len(settings)):
                w,n=settings[index]
                start=perf_counter()
                m.sample(n,route,True,w,102121000+int(route)*100+int(index),n*rep)
                if rep:times[w,n].append((perf_counter()-start)*1e6/n)
        best=min(settings,key=lambda s:np.median(times[s]))
        hold=[best] if best==(16,8192) else [best,(16,8192)]
        result={s:[] for s in hold}
        for i in range(11):
            for k in rng.permutation(len(hold)):
                w,n=hold[k]
                start=perf_counter()
                m.sample(n,route,True,w,102122000+10*int(route)+int(k),n*i)
                result[w,n].append((perf_counter()-start)*1e6/n)
        check,c=m.sample(8192,route,True,16,102123000+route)
        other,d=m.sample(8192,route,True,best[0],102123000+route)
        assert np.array_equal(check,other) and np.array_equal(c,d)
        rows.append(dict(pnr=route,chosen=list(best),pilot=[dict(workers=w,n=n,us=t) for (w,n),t in times.items()],
            hold=[dict(workers=w,n=n,us=t,median_us=float(np.median(t)),quartiles=np.percentile(t,[25,75]).tolist())
                  for (w,n),t in result.items()]))
    save(out,'summary.json',dict(logical_cpus=physical,cold_first_2048_us=cold,rows=rows))
    manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):
        main(ap.parse_args().output.resolve())
