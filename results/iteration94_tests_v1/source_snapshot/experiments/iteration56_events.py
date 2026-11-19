"""Extract sparse cylinder events and certify the final-bit split gain."""
import sys,json,gzip,hashlib,shutil,time
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.prefix_event_witness import sparse_events,prefix_split_witness
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration56_events';out.mkdir(exist_ok=False)
    names=['research/iteration56_prefix_plan.md','src/gbslab/prefix_event_witness.py',*[f'experiments/{n}.py' for n in ('iteration56_events','iteration22_external','iteration33_run','verify_iteration34')]]
    hashes={n:digest(ROOT/n) for n in names};deps={};rows=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):
        deps[n]=digest(ROOT/n);b=(ROOT/n).read_bytes();return decode(json.loads(gzip.decompress(b) if n.endswith('.gz') else b))
    old=read('results/iteration53_prefix_validation/bounds.json');manifest=read('results/iteration53_manifest.json');assert deps['results/iteration53_prefix_validation/bounds.json']==manifest['artifacts']['results/iteration53_prefix_validation/bounds.json']
    physical=read('results/iteration56_target/target.json');p=physical['law']['intervals'];transfer=physical['transfer_upper'];assert len(p)==65536
    for k in range(1,5):
        tick=time.perf_counter();r=read(f'results/iteration56_prefix/global_{k}.audit.json.gz');q=r['laws']['uint64']['distribution'];assert len(q)==65536
        events=sparse_events(q,p);split=prefix_split_witness(q,p)
        before=next(a for a in old if a['cutoff']==k)['prefix_TV_intervals']['uint64']
        fine=(sum(max(F(0),a-x,x-b) for x,(a,b) in zip(q,p))/2,sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(q,p))/2)
        propagated=(before[0]+split['gain_interval'][0],before[1]+split['gain_interval'][1])
        combined=(max(fine[0],propagated[0]),min(fine[1],propagated[1]));assert combined[0]<=combined[1]
        top=sorted(split['rows'],key=lambda r:(-r['gain_interval'][0],r['parent']))[:16]
        raw=(json.dumps(encode(split),indent=2)+'\n').encode();data=gzip.compress(raw,compresslevel=6,mtime=0);name=f'global_{k}.split.json.gz';(out/name).write_bytes(data)
        save(out/f'global_{k}.events.json',encode(events))
        row=dict(cutoff=k,direct_prefix_TV=fine,old_prefix_TV=before,gain_interval=split['gain_interval'],propagated_prefix_TV=propagated,combined_prefix_TV=combined,full144_TV_lower=combined[0],shifted_full144_TV_lower=max(F(0),combined[0]-transfer),certified_opposite_parents=split['certified_opposite_parents'],top_gain_parents=top,storage=dict(path=name,raw_bytes=len(raw),compressed_bytes=len(data),raw_sha256=hashlib.sha256(raw).hexdigest()),seconds=time.perf_counter()-tick)
        rows.append(row);save(out/'rows.json',encode(rows));print('events',k,'lower',float(combined[0]),'opposite',split['certified_opposite_parents'],flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items())
    save(out/'summary.json',dict(completed=True,cutoffs=4,parents_per_cutoff=32768,events_per_cutoff=16,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True))


if __name__=='__main__':main()
