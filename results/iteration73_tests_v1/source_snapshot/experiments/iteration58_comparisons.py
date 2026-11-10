"""Frozen candidate-budget comparisons, without claiming new sampler accuracy."""
import sys,json,shutil,math
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration58_comparisons';out.mkdir(exist_ok=False);deps={};checks=[];names=[f'experiments/{n}.py' for n in ('iteration58_comparisons','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def ck(name,ok):checks.append(dict(name=name,passed=bool(ok)));assert ok,name
    v=read('results/iteration58_coupling_validation/validation.json');ck('all independent checks passed',v['count']==v['passed']);rows=read('results/iteration58_coupling/retrospective_rows.json');rank4=read('results/iteration58_coupling/rank4_rows.json')
    for n in deps:
        if n.endswith('_rows.json'):ck('validated bytes '+n,deps[n]==v['dependencies'][n])
    for r in rows:
        tag=f"case{r['case']}";ck(tag+' node count',r['nodes']==math.prod(n+2 for n in r['job']['cells']));ck(tag+' entire program covered',r['actual_TV'][1]<=r['new_total_upper']<r['old_total_upper']);ck(tag+' ideal error covered',r['ideal_grid_TV'][1]<=r['bound']['best_upper']);ck(tag+' L1 improves old',r['bound']['exact_l1_upper']<r['bound']['old_axis_upper'])
    two=[]
    for start in range(0,24,4):
        group=rows[start:start+4];fine=group[3];tag=f'group{start//4}';ck(tag+' complete fixed candidates',[r['job']['cells'] for r in group]==[[4,4],[16,4],[4,16],[16,16]]);eligible=[r for r in group if r['nodes']==108];old=min(eligible,key=lambda r:(r['old_total_upper'],r['case']));new=min(eligible,key=lambda r:(r['new_total_upper'],r['case']));passing=[r for r in group if r['new_total_upper']<=fine['old_total_upper']];cost=min(r['nodes'] for r in passing);selected=min((r for r in passing if r['nodes']==cost),key=lambda r:(r['new_total_upper'],r['case']));l1passing=[r for r in group if min(F(1),r['other_errors_upper']+r['bound']['exact_l1_upper'])<=fine['old_total_upper']]
        ck(tag+' min-cost witness',selected['new_total_upper']<=fine['old_total_upper'] and all(r['new_total_upper']>fine['old_total_upper'] for r in group if r['nodes']<cost));ck(tag+' fixed budget minimizer',all(new['new_total_upper']<=r['new_total_upper'] for r in eligible));two.append(dict(family=fine['job']['family'],dimension=fine['job']['dimension'],fixed_budget=108,old_selected_case=old['case'],new_selected_case=new['case'],selected_actual_TV=new['actual_TV'],alternative_actual_TV=next(r['actual_TV'] for r in eligible if r['case']!=new['case']),selected_is_best_actual=new['actual_TV'][1]<next(r['actual_TV'][0] for r in eligible if r['case']!=new['case']),old_fine_threshold=fine['old_total_upper'],old_fine_nodes=324,new_min_nodes=cost,new_min_case=selected['case'],new_min_upper=selected['new_total_upper'],exact_l1_min_nodes=min(r['nodes'] for r in l1passing),node_ratio=F(324,cost)))
    four=[]
    for fam in ('real_pair','complex_rotated'):
        group=[r for r in rank4 if r['family']==fam];fine=group[15];threshold=fine['bound']['old_axis_upper'];eligible=[r for r in group if r['nodes']<=11664];old=min(eligible,key=lambda r:(r['bound']['old_axis_upper'],r['index']));new=min(eligible,key=lambda r:(r['bound']['best_upper'],r['index']));passing=[r for r in group if r['bound']['best_upper']<=threshold];cost=min(r['nodes'] for r in passing);selected=min((r for r in passing if r['nodes']==cost),key=lambda r:(r['bound']['best_upper'],r['index']));l1cost=min(r['nodes'] for r in group if r['bound']['exact_l1_upper']<=threshold)
        for r in group:ck(fam+str(r['index'])+' nodes and bounds',r['nodes']==math.prod(n+2 for n in r['cells']) and r['bound']['best_upper']<r['bound']['old_axis_upper'])
        ck(fam+' fixed budget minimizer',all(new['bound']['best_upper']<=r['bound']['best_upper'] for r in eligible));ck(fam+' cost minimum',all(r['bound']['best_upper']>threshold for r in group if r['nodes']<cost));four.append(dict(family=fam,fixed_budget=11664,old_selected_index=old['index'],new_selected_index=new['index'],selected_cells=new['cells'],old_selected_upper=old['bound']['old_axis_upper'],new_selected_upper=new['bound']['best_upper'],old_fine_threshold=threshold,old_fine_nodes=fine['nodes'],new_min_nodes=cost,new_min_index=selected['index'],new_min_upper=selected['bound']['best_upper'],exact_l1_min_nodes=l1cost,node_ratio=F(fine['nodes'],cost)))
    def counts(rr):return dict(total=len(rr),strictly_improved=sum(r['bound']['best_upper']<r['bound']['old_axis_upper'] for r in rr),joint_wins=sum(r['bound']['joint_curved_upper']<r['bound']['exact_l1_upper'] for r in rr),exact_l1_wins=sum(r['bound']['exact_l1_upper']<r['bound']['joint_curved_upper'] for r in rr),ties=sum(r['bound']['joint_curved_upper']==r['bound']['exact_l1_upper'] for r in rr),relative_reduction_range=[min(1-r['bound']['best_upper']/r['bound']['old_axis_upper'] for r in rr),max(1-r['bound']['best_upper']/r['bound']['old_axis_upper'] for r in rr)])
    save(out/'resources.json',encode(dict(two_drive=two,rank4=four)));ck('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));ck('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),source_hashes=hashes,dependencies=deps));save(out/'summary.json',encode(dict(completed=True,retrospective=counts(rows),rank4=counts(rank4),two_drive_groups=len(two),rank4_groups=len(four),two_drive_actual_choice_better=sum(g['selected_is_best_actual'] for g in two),scope='Candidate node counts and certified bounds, not runtime speedups; two-drive laws unchanged, rank4 ideal-channel only.')));print(json.dumps(dict(checks=len(checks),two_drive=counts(rows)['joint_wins'],rank4=counts(rank4)['joint_wins'])))

if __name__=='__main__':main()
