"""Fresh paired double-spectrum records, using the two frozen column orders."""
import argparse
import json
from iteration94_common_v2 import ROOT, Path, np, H, load93, sample93, save, manifest

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
out = parser.parse_args().output.resolve()
out.mkdir(parents=True, exist_ok=False)
inputs = [ROOT/f'results/iteration94_actual_quality_v1/input_{j}.npz' for j in (0, 1)]
orders = [ROOT/f'results/iteration95_order_pilot_v1/actual{j}_order.json' for j in (0, 1)]
sources = [Path(__file__), ROOT/'results/iteration94_build_v2/mps.dll', *inputs, *orders]
experiment = 0x95060000
save(out, 'plan.json', dict(records=12, batch=4, experiment=experiment, budgets=[.1,.01],
    threads=4, bond=32, d=6, cap=256, pair_by='experiment, sector, global record and purpose',
    primary='paired whole-record completion; exact two-sided McNemar sign test on discordant pairs',
    hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources}))
lib, runtime = load93()
save(out, 'backend.json', lib._backend)
success = np.ones((2, 12), dtype=bool)
for j, budget in enumerate((.1,.01)):
    with np.load(inputs[j]) as f:
        z = dict(f)
    perm = np.asarray(json.loads(orders[j].read_text())['order'])
    assert np.array_equal(np.sort(perm), np.arange(len(perm)))
    moved = dict(u=z['u'][:,perm], t=z['t'][perm], alpha=z['alpha'][perm],
                 noise=z['noise'][np.r_[perm,len(perm)+perm]])
    for off in range(0,12,4):
        for k in ((0,1) if (off//4+j)%2==0 else (1,0)):
            model = (z,moved)[k]
            raw, meta = sample93(lib, **model, N=4, experiment=experiment, sector=j,
                offset=off, threads=4, d=6, bond=32, budget=budget, cap=256)
            np.savez_compressed(out/f'raw_{j}_{off}_{k}.npz', **raw)
            save(out, f'batch_{j}_{off}_{k}.json', meta)
            assert meta['returncode']==0 and np.isin(raw['errors'],[0,2]).all()
            success[k,off:off+4] &= raw['errors']==0
            print(j,off,k,meta['failures'],flush=True)
from scipy.stats import binomtest
gained = int(np.count_nonzero(success[1] & ~success[0]))
lost = int(np.count_nonzero(success[0] & ~success[1]))
save(out,'summary.json',dict(completed=success.sum(1).tolist(),records=12,gained=gained,lost=lost,
    p=float(binomtest(gained,gained+lost).pvalue) if gained+lost else 1.))
manifest(out)
