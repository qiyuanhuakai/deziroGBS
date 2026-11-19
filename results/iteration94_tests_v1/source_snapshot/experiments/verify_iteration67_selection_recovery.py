"""Independent exhaustive minimization over the frozen certified candidates."""
from iteration67_support import ROOT,save,read,digest,snapshot,graphs,EPSILONS
from iteration63_common import encode

def main():
    out=ROOT/'results/iteration67_selection_validation';hashes=snapshot(out);deps={};checks=[]
    def load(n):deps[n]=digest(ROOT/n);return read(ROOT/n)
    def ck(n,ok):checks.append(dict(name=n,passed=bool(ok)));assert ok,n
    v=load('results/iteration67_costs_recovery_validation/validation.json');rows=load('results/iteration67_costs_recovery_validation/candidates.json');plans=load('results/iteration67_selection/plans.json');s=load('results/iteration67_selection/summary.json');ck('bound candidate source',v['count']==v['passed'] and deps['results/iteration67_costs_recovery_validation/candidates.json']==s['dependencies']['results/iteration67_costs_recovery_validation/candidates.json']);ck('ten fixed requests',len(plans)==10)
    for index,p in enumerate(plans):
        g=graphs()[index//2];epsilon=EPSILONS[index%2];ck(f'plan{index} identity',p['index']==index and encode(p['graph'])==encode(g) and p['epsilon']==epsilon);qualifying=[r for r in rows if r['graph_index']==index//2 and r['conditional_TV']<=epsilon];ck(f'plan{index} qualified coverage',p['feasible_candidates']==len(qualifying))
        for route in ('joint','fixed_alpha'):
            subset=[r for r in qualifying if route=='joint' or r['alpha']==4*g['n']*g['n']];chosen=p[route];ck(f'plan{index} {route} feasibility',(chosen is None)==(not subset))
            if chosen is None:continue
            ck(f'plan{index} {route} membership',chosen in subset)
            rank=lambda r:(r['expected_words'],r['expected_steps'],r['alpha'],r['steps'])
            for j,r in enumerate(subset):ck(f'plan{index} {route} minimality {j}',rank(chosen)<=rank(r))
        if p['joint'] is not None and p['fixed_alpha'] is not None:ck(f'plan{index} cost dominance',p['joint']['expected_words']<=p['fixed_alpha']['expected_words'])
    ck('sources unchanged',all(digest(ROOT/n)==h for n,h in hashes.items()));ck('dependencies unchanged',all(digest(ROOT/n)==h for n,h in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),plans=10,routes=sum(p[k] is not None for p in plans for k in ('joint','fixed_alpha')),source_hashes=hashes,dependencies=deps));print('selection verified',len(checks),flush=True)
if __name__=='__main__':main()
