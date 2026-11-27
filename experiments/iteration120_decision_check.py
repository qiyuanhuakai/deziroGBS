"""Check the fresh decisions closest to each computed threshold under a second conditioning."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from iteration120_replay import OUT,init,stable,clamps

def initialize(path):
    init(path);clamps.stable=partial(stable,compensated=True)

def run():
    summary=[]
    for r in range(2):
        source=json.loads((OUT/f'full_{r}_256.json').read_bytes())['decisions']
        pool_rows=[]
        for stages in (1,2):
            def margin(x):
                t=x['threshold'];m=x['weights'][0]
                thresholds=[.8*m,1.25*m]+x['weights'][1:]
                return min(abs(t-v) for v in thresholds)
            rows=[x for x in source if x['stages']==stages]
            pool_rows+=sorted(rows,key=margin)[:4]
        with ProcessPoolExecutor(2,initializer=initialize,initargs=(OUT/f'base_{r}_256.npz',)) as pool:
            after=list(pool.map(clamps.process,[(r,x['index'],False,1.6,120) for x in pool_rows]))
        for a,b in zip(pool_rows,after):
            assert a['stages']==b['stages'] and a['accepted']==b['accepted']
            summary.append(dict(direct=bool(r),index=a['index'],stages=a['stages'],accepted=a['accepted'],
                                threshold=a['threshold'],reference=a['weights'],compensated=b['weights']))
    (OUT/'decision_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(len(summary),'near-boundary decisions verified.')
if __name__=='__main__':run()
