"""Check archive provenance, full distribution comparisons and both user-facing routes."""
import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from threadpoolctl import threadpool_limits
from iteration96_group_common_v1 import ROOT,read,save,manifest,H,np
from iteration96_sample_v1 import run
from iteration96_group_benchmark_v1 import fixture
base=ROOT/'results'
p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
dirs=['tree_build_v1','frugal_build_v1','optical_v1','frugal16_v1','frugal16_v2','component_v1','herald_v1','pe16_v1','identify_v1','quad_v1','calibrated_v1']
sub=lambda s:base/('iteration96_'+s)
files=[Path(__file__),ROOT/'experiments/iteration96_sample_v1.py',*(sub(n)/'manifest.json' for n in dirs)]
save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},
    basis='independent raw evidence, CLI samples and cold input-to-first-batch wall time; zero-mean 16-port reduced input'))
hash_count=0;plan_count=0
for name in dirs:
    d=sub(name);entries=read(d/'manifest.json')
    assert set(entries)=={f.name for f in d.iterdir() if f.name!='manifest.json'}
    for k,sha in entries.items():assert H(d/k)==sha,(name,k)
    for k,sha in read(d/'plan.json')['hashes'].items():
        path=ROOT/k
        if name=='frugal16_v1' and k=='experiments/iteration96_frugal_benchmark_v1.py':path=d/'original_driver.py'
        assert H(path)==sha,(name,k)
        plan_count+=1
    hash_count+=len(entries)
hardware=read(sub('calibrated_v1')/'summary.json')
assert all(x['records']==262144 for x in hardware)
pe=read(sub('pe16_v1')/'summary.json')
assert all(x['records']==16384 and x['failures']==0 for x in pe)
oracle=read(sub('quad_v1')/'summary.json')
assert len(oracle)==4 and max(x['direct'] for x in oracle)<3e-12
optical=sub('optical_v1');cli=ROOT/'experiments/iteration96_sample_v1.py'
for route in ['click','pnr']:
    path=out/f'cli_{route}.npz';cmd=[sys.executable,str(cli),str(optical/'mixed_reference.npz'),str(path),'--route',route,
        '--seed','96081000','--shots','512','--workers','4']
    proc=subprocess.run(cmd,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    with np.load(optical/'mixed_reference.npz') as z, np.load(path) as sample:
        replay=run(z,route,384,96081000,128,1)
        assert not sample['status'].any() and np.array_equal(sample['records'][128:],replay['records'])
        assert sample['method']==('component_click' if route=='click' else 'herald')
with threadpool_limits(limits=1,user_api='blas'):
    z=np.load(optical/'actual16_reference.npz');cold=[]
    for rep in range(5):
        start=perf_counter();r=run(z,'click',2048,96082000+rep,0,1);cold.append(1e6*(perf_counter()-start)/2048)
    q,mu,_,_=fixture('actual1152');large=dict(q=q,mu=mu)
    for route in ['click','pnr']:
        r=run(large,route,16,96083000,0,1)
        assert r['method']=='tree_'+route and r['records'].shape==(16,1152) and not r['status'].any()
    pure=np.load(optical/'pure4_reference.npz')
    for route in ['click','pnr']:
        small=run(dict(barg=pure['barg']),route,64,96083100,0,1)
        assert small['records'].shape==(64,4) and not small['status'].any()
    for route in ['click','pnr']:
        vacuum=run(dict(barg=np.zeros((3,3),complex)),route,256,96083200,0,1)
        assert not vacuum['records'].any() and not vacuum['status'].any()
    thermal=run(dict(q=11*np.eye(2),mu=np.zeros(2)),'pnr',131072,96083300,0,4)
    tail=thermal['tail'];counts=thermal['records'][:,0]
    assert np.array_equal(tail>0,counts==33)
    expected=(10/11)**33
    assert abs(tail.mean()-expected)<.003
save(out,'validation.json',dict(passed=True,archive_files_checked=hash_count,input_hashes_checked=plan_count,
    hardware_attempts=[r['records'] for r in hardware],thermal_tail_predicted=expected,thermal_tail_observed=float(tail.mean()),
    actual16_input_to_complete_2048_us=cold,actual16_median_us=float(np.median(cold)),quad_max_direct_tv=max(r['direct'] for r in oracle)))
manifest(out)
