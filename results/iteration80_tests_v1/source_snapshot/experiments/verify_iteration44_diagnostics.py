"""Independent order permutation, event witness, and exceptional-prefix audit."""
import json,shutil,math
from collections import defaultdict
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration41_independent import word_probability
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/iteration44_diagnostics';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration44_diagnostics','iteration22_external','iteration33_run','verify_iteration34','iteration41_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];orders=[];infinities=[];components=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    cases=[read(f'results/iteration44_panel/case_{i:02d}.json') for i in range(24)];native={(i,k):read(f'results/iteration44_native/case_{i:02d}_k{k}.audit.json') for i in range(24) for k in (2,4)};reported=read('results/iteration44_analysis/order_comparisons.json');summary=read('results/iteration44_analysis/summary.json');main=read('results/iteration44_recovery_v2/validation.json');check('main verified',main['count']==main['passed']==113573)
    def fingerprint(tree):
        coeff=defaultdict(F)
        for i,j,score in tree['selected_edges']:
            for p in tree['pairs'][str((i,j))]:coeff[p]+=p
            for p in tree['singles'][i]+tree['singles'][j]:coeff[p]-=p
        return {p:c for p,c in coeff.items() if c}
    for pair_index,i in enumerate(range(0,24,2)):
        n=cases[i]['job']['modes'];check(f'pair{i} orders',cases[i]['job']['order']==list(range(n)) and cases[i+1]['job']['order']==list(reversed(range(n))));perm=lambda x:(x>>n<<n)|int(format(x&((1<<n)-1),f'0{n}b')[::-1],2);result=dict(cases=(i,i+1))
        check(f'pair{i} physical target',all(cases[i]['target']['target_intervals'][a]==cases[i+1]['target']['target_intervals'][perm(a)] for a in range(1<<n)))
        for key in ('completion','tree','K2','K4'):
            def law(j):return cases[j]['projection']['law'] if key=='completion' else cases[j]['tree']['law'] if key=='tree' else native[j,int(key[1:])]['laws']['uint64']['distribution']
            left,right=law(i),law(i+1);tv=sum(abs(left[a]-right[perm(a)]) for a in range(len(left)))/2;check(f'pair{i} {key}',tv==reported[pair_index][key+'_aligned_TV']);result[key+'_aligned_TV']=tv
        score_equal=fingerprint(cases[i]['tree'])==fingerprint(cases[i+1]['tree']);result['tree_selected_score_exactly_equal']=score_equal;result['tree_score_log_coefficients_forward']=[dict(probability=p,coefficient=c) for p,c in sorted(fingerprint(cases[i]['tree']).items())];result['tree_score_log_coefficients_reverse']=[dict(probability=p,coefficient=c) for p,c in sorted(fingerprint(cases[i+1]['tree']).items())];orders.append(result)
    for i,c in enumerate(cases):
        V=c['physical']['covariance'];n=c['job']['modes'];pending=set(range(n));groups=[]
        while pending:
            group=[min(pending)];pending.remove(group[0])
            for a in group:
                for b in sorted(pending.copy()):
                    if any(V[x][y]!=0 for x in (a,a+n) for y in (b,b+n)):group.append(b);pending.remove(b)
            groups.append(sorted(group))
        components.append(dict(case=i,covariance_mode_components=groups))
        for cutoff in (2,4):
            raw=read(f'results/iteration44_native/case_{i:02d}_k{cutoff}.stdout.json');table={x['prefix']:float.fromhex(x['threshold_hex']) for x in raw['prefix_tree']};law=native[i,cutoff]['laws']['uint64']['distribution']
            for prefix,q in table.items():
                if not math.isinf(q):continue
                mass=F(1)
                for j,bit in enumerate(prefix):p=word_probability(table[prefix[:j]]);mass*=p if bit=='0' else 1-p
                selected=[a for a in range(1024) if all(str((a>>j)&1)==bit for j,bit in enumerate(prefix))];check(f'infinity {i}:{cutoff}:{prefix}',sum(law[a] for a in selected)==mass);infinities.append(dict(case=i,cutoff=cutoff,prefix=prefix,mass=mass,threshold='positive infinity' if q>0 else 'negative infinity'))
    check('infinity summary',len(infinities)==summary['infinite_threshold_nodes'] and sum(r['mass']>0 for r in infinities)==summary['positive_finite_word_prefix_mass_infinite_nodes'])
    event=read('results/iteration44_analysis/event_witness.json');r=cases[0];selected=event['masks'];Q=sum(r['projection']['law'][i] for i in selected);P=(sum(r['target']['target_intervals'][i][0] for i in selected),sum(r['target']['target_intervals'][i][1] for i in selected));gap=Q-r['bridge']['TV_to_true_maxent_upper']-P[1];control=read('results/iteration44_mechanism/case_00.json');check('explicit event proves counterexample',Q==event['completion_probability'] and P==tuple(event['physical_probability_interval']) and gap==event['true_maxent_event_gap_lower']>control['true_feasible_physical_TV_upper']==event['same_moment_K2_feasible_upper'])
    check('unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'order_details.json',encode(orders));save(out/'infinite_prefixes.json',encode(infinities));save(out/'covariance_components.json',components);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),order_pairs=12,exact_equal_selected_tree_scores=sum(r['tree_selected_score_exactly_equal'] for r in orders),different_laws_with_equal_tree_score=sum(r['tree_selected_score_exactly_equal'] and r['tree_aligned_TV']>0 for r in orders),infinite_nodes=len(infinities),positive_mass_infinite_nodes=sum(r['mass']>0 for r in infinities),maximum_infinite_prefix_mass=float(max(r['mass'] for r in infinities)),source_hashes=hashes,dependencies=deps));print('diagnostics verified',len(checks),'equal tree scores',sum(r['tree_selected_score_exactly_equal'] for r in orders),flush=True)
if __name__=='__main__':main()
