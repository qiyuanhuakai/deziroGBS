"""Choose only after independent law/cost certification; keep infeasibility."""
from iteration67_support import ROOT,save,read,digest,snapshot,graphs,EPSILONS

def main():
    out=ROOT/'results/iteration67_selection';hashes=snapshot(out);deps={}
    def load(n):deps[n]=digest(ROOT/n);return read(ROOT/n)
    v=load('results/iteration67_costs_validation/validation.json');assert v['count']==v['passed'] and v['cases']==20;rows=load('results/iteration67_costs_validation/candidates.json');plans=[]
    def key(r):return r['expected_words'],r['expected_steps'],r['alpha'],r['steps']
    for gi,g in enumerate(graphs()):
        for epsilon in EPSILONS:
            candidates=[r for r in rows if r['graph_index']==gi and r['conditional_TV']<=epsilon];base=[r for r in candidates if r['alpha']==4*g['n']*g['n']];plans.append(dict(index=len(plans),graph=g,epsilon=epsilon,joint=min(candidates,key=key) if candidates else None,fixed_alpha=min(base,key=key) if base else None,feasible_candidates=len(candidates)))
    save(out/'plans.json',plans);assert all(digest(ROOT/n)==h for n,h in hashes.items()) and all(digest(ROOT/n)==h for n,h in deps.items());save(out/'summary.json',dict(completed=True,plans=10,routes=sum(p[k] is not None for p in plans for k in ('joint','fixed_alpha')),source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True));print('selected certified routes',sum(p[k] is not None for p in plans for k in ('joint','fixed_alpha')),flush=True)
if __name__=='__main__':main()
