"""Measure each scoring route separately and replay with a second numerical conditioning."""
import json
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from iteration119_clamps import OUT,initialize,process

def run():
    path=OUT/'exact_bands_8.npz'
    base=json.loads((OUT/'exact_bands_8.json').read_bytes())
    result=[]
    for r in range(2):
        # A deterministic prefix for an independent precision check using a new gauge.
        start=perf_counter()
        with ProcessPoolExecutor(5,initializer=initialize,initargs=(path,{})) as pool:
            rows=list(pool.map(process,[(r,i,False,1.6) for i in range(8)]))
        elapsed=perf_counter()-start
        accepted=sum(x['accepted'] for x in rows)
        result.append(dict(direct=bool(r),candidates=len(rows),accepted=sum(x['accepted'] for x in rows),
                           score_wall_seconds=elapsed,workers=5,conditioning_threshold=1.6,
                           base_wall_seconds=base[r]['wall_s'],
                           seconds_per_accepted=(elapsed+base[r]['wall_s'])/accepted if accepted else None,
                           replayed=rows))
    (OUT/'timing_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print([(x['accepted'],x['score_wall_seconds'],x['seconds_per_accepted']) for x in result],flush=True)

if __name__=='__main__':run()
