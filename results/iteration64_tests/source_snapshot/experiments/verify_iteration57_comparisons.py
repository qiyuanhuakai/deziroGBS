"""Independent selected-overlap identity, complementary mass and bound audit."""
import sys,json,shutil,gzip,math
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration57_comparisons';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration57_comparisons','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);b=(ROOT/n).read_bytes();return decode(json.loads(gzip.decompress(b) if n.endswith('.gz') else b))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    pv=read('results/iteration57_paths_validation/validation.json');tv=read('results/iteration57_target_validation/validation.json');check('audits complete',pv['count']==pv['passed'] and tv['count']==tv['passed']);check('complete coverage',pv['complete_paths']==1044 and pv['path_steps']==13120 and pv['public_steps']==12362 and pv['local_full_atoms']==1024 and tv['physical_certificates']==7 and tv['complete_outputs']==5)
    summary=read('results/iteration57_overlap/summary.json');check('overlap complete',summary['completed'] and summary['jobs']==5 and summary['global_outputs_per_cutoff']==5 and summary['local_outputs']==1024)
    for n,s in summary['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/'results/iteration57_overlap/source_snapshot'/n))
    for n,s in summary['dependencies'].items():check('dependency '+n,digest(ROOT/n)==s);deps[n]=s
    physical=read('results/iteration57_target/target.json');check('validated physical output',deps['results/iteration57_target/target.json']==tv['dependencies']['results/iteration57_target/target.json']);oldlocal=read('results/iteration37_local/target_000.json');oldlocal_audit=read('results/iteration39_local/case_03.audit.json');oldglobal=read('results/iteration56_prefix_validation/bounds.json');oldstrong=read('results/iteration56_events_validation/rows.json');produced=read('results/iteration57_overlap/rows.json');check('five labels',[r['label'] for r in produced]==['local_4','global_1','global_2','global_3','global_4'])
    for row in produced:
        label=row['label'];n=row['modes'];k=row['cutoff'];tag=label+' ';path='results/iteration57_paths/'+label+'.audit.json';audit=read(path);check(tag+'validated program law',deps[path]==pv['dependencies'][path]);atoms=read('results/iteration57_overlap/'+label+'.atoms.json');target=oldlocal['law']['intervals'] if n==10 else [r['probability_interval'] for r in physical['targets']];check(tag+'selected count',len(atoms)==row['selected_outputs']==(1024 if n==10 else 5));check(tag+'distinct complete outputs',len({r['output'] for r in atoms})==len(atoms) and all(len(r['output'])==n for r in atoms))
        for i,(a,r,p) in enumerate(zip(atoms,audit['laws'],target)):check(tag+f'atom{i} binding',a['output']==r['output'] and a['target_interval']==p and a['probabilities']=={model:r['models'][model]['probability'] for model in ('uint64','continuous')})
        diagnostics={}
        for model in ('uint64','continuous'):
            q=[a['probabilities'][model] for a in atoms];m=row['models'][model];bound=m['overlap'];A=sum(a for a,b in target);B=sum(b for a,b in target);Q=sum(q)
            # min(x,y)=(x+y-|x-y|)/2, separate from production min loops.
            ol=sum((x+a-abs(x-a))/2 for x,(a,b) in zip(q,target));oh=sum((x+b-abs(x-b))/2 for x,(a,b) in zip(q,target));restq=1-Q;restp=(max(F(0),1-B),1-A);lo=max(F(0),1-oh-min(restq,restp[1]));hi=1-ol
            check(tag+model+' overlap endpoints',tuple(bound['selected_overlap_interval'])==(ol,oh) and bound['upper_gap_from_one']==ol and tuple(bound['full_TV_interval'])==(lo,hi) and 0<=lo<=hi<=1)
            check(tag+model+' complements',bound['program_selected_mass']==Q and tuple(bound['target_selected_mass_interval'])==(A,B) and bound['program_remaining_mass']==restq and tuple(bound['target_remaining_mass_interval'])==restp)
            if n==10:
                direct=(sum(max(F(0),a-x,x-b) for x,(a,b) in zip(q,target))/2,sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(q,target))/2);prior=oldlocal_audit['TV_intervals'][model];check(tag+model+' old complete law and distance',q==oldlocal_audit['laws'][model]['distribution'] and sum(q)==1 and tuple(prior)==direct)
            else:prior=(next(r for r in oldstrong if r['cutoff']==k)['full144_lower'] if model=='uint64' else next(r for r in oldglobal if r['cutoff']==k)['prefix_TV_intervals'][model][0],F(1))
            combined=(max(prior[0],lo),min(prior[1],hi));check(tag+model+' prior intersection',tuple(m['prior_interval'])==tuple(prior) and tuple(m['combined_full_TV_interval'])==combined and combined[0]<=combined[1])
            if n==144:check(tag+model+' shifted triangle transfer',tuple(m['shifted_full_TV_interval'])==(max(F(0),combined[0]-physical['transfer_upper']),min(F(1),combined[1]+physical['transfer_upper'])))
            diagnostics[model]=dict(gap_from_one=ol,TV_lower=combined[0],TV_upper=combined[1],selected_program_mass=Q,selected_target_interval=(A,B),shifted_upper=m['shifted_full_TV_interval'][1] if n==144 else None,strict_nontrivial_upper=combined[1]<1)
        raw=read('results/iteration57_paths/'+label+'.stdout.json.gz');nonfinite=zero_threshold=unreachable=0
        for a,r in zip(audit['laws'],raw['path_probes']):
            mass=F(1)
            for step,factor in zip(r['steps'],a['models']['uint64']['factors']):
                threshold=float.fromhex(step[0]);nonfinite+=bool(mass and not math.isfinite(threshold));zero_threshold+=bool(mass and threshold==0);mass*=factor
            unreachable+=mass==0
        rows.append(dict(label=label,modes=n,cutoff=k,selected_outputs=len(atoms),models=diagnostics,reachable_nonfinite_steps=nonfinite,reachable_zero_threshold_steps=zero_threshold,zero_mass_outputs=unreachable))
    native=read('results/iteration57_paths/rows.json');check('all native completed',len(native)==5 and all(r['status']=='completed' for r in native));check('native accounting',sum(r['paths'] for r in native)==1044 and sum(r['steps'] for r in native)==13120 and sum(r['probe_control_checks'] for r in native)==1124 and sum(r['public_path_checks'] for r in native)==12362)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=len(checks),source_hashes=hashes,dependencies=deps));save(out/'summary.json',dict(completed=True,jobs=5,global_cutoffs=4,complete_paths=1044,path_steps=13120,physical_certificates=7,local_atoms=1024,strict_global_word_uppers=sum(r['models']['uint64']['strict_nontrivial_upper'] for r in rows if r['modes']==144),shifted_uppers_equal_one=sum(r['models']['uint64']['shifted_upper']==1 for r in rows if r['modes']==144),scope='Selected complete-output overlap certifies full-law upper bounds. Bounds close to one are not useful sampling-accuracy guarantees.'))
    print('comparison checks',len(checks),flush=True)


if __name__=='__main__':main()
