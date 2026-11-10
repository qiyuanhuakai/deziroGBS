"""Recover only a preserved integer-to-decimal serialization failure, unchanged science."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,shutil,sys
from pathlib import Path
from fractions import Fraction as F
from time import perf_counter
import numpy as np
from iteration34_run import ROOT,patterns,physical_covariance,analytic_interval,collect_native,compact,VacuumEnclosure,selected_atoms
from iteration22_external import digest,save
from iteration33_run import encode

def main():
    # Trusted, locally generated exact Fractions can exceed Python's default
    # decimal-digit limit. This changes serialization only, not arithmetic.
    sys.set_int_max_str_digits(0)
    original=ROOT/'results/iteration34_panel';failed=json.loads((original/'case_37.json').read_text());assert failed['status']=='failed' and 'integer string conversion' in failed['traceback']
    plan=json.loads((original/'plan.json').read_text());job=plan['jobs'][37];assert job==dict(panel='large',modes=144,family='thermal',exponent=40,max_clicks=1,precision='auto')
    out=ROOT/'results/iteration34_recovery';out.mkdir(exist_ok=False);names=['experiments/iteration34_recover.py','experiments/iteration34_run.py','experiments/iteration22_external.py','experiments/iteration33_run.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    paths=['results/iteration34_panel/case_37.json','results/iteration34_panel/case_37.npz','results/iteration34_panel/plan.json','results/iteration31_build/libwire_forest.dll','results/iteration31_build/build.json'];dependencies={n:digest(ROOT/n) for n in paths};save(out/'plan.json',dict(original_case=37,job=job,source_hashes=hashes,dependencies=dependencies,integer_string_digit_limit=sys.get_int_max_str_digits(),reason='Retry only decimal serialization failure. All probabilities, precision policy and selected paths unchanged. Original failed output retained.'))
    started=perf_counter();row=dict(sequence=37,**job);chosen=patterns(144,1);V=physical_covariance(job);assert np.array_equal(V,np.load(original/'case_37.npz')['covariance']);begin=perf_counter();cert=VacuumEnclosure(V);row['certificate_seconds']=perf_counter()-begin;cert_atoms={c:cert.atom(c) for c in chosen};by_count={k:analytic_interval(job,k) for k in range(2)};target={c:by_count[len(c)] for c in chosen}
    for c,(lo,hi) in cert_atoms.items():assert lo<=target[c][0]<=target[c][1]<=hi
    np.savez_compressed(out/'case_37.npz',covariance=V,R=cert.R);row['target_scope']='physically valid analytic thermal target';begin=perf_counter();table,info=collect_native(V,job,chosen,ROOT/paths[-2]);row.update(info);row['native_seconds']=perf_counter()-begin;audits=[]
    def oracle(h):
        value=table[h]['q_hex'];return None if value is None else F(float.fromhex(value))
    for bits in (None,53):
        for k in (0,1):
            r=selected_atoms(144,[c for c in chosen if len(c)<=k],oracle,lambda c:target[c],uniform_bits=bits);audits.append(dict(max_clicks=k,**compact(r)))
    output=dict(job=job,patterns=chosen,target_intervals=[target[c] for c in chosen],oracle=list(table.values()),gaussian_certificate=dict(diagnostics=cert.diagnostics,atoms=[dict(pattern=c,interval=b) for c,b in cert_atoms.items()]),audits=audits);save(out/'case_37.audit.json',encode(output));row.update(status='completed',unique_conditionals=len(table),selected_atoms=len(chosen),failure_nodes=sum(r['q_hex'] is None for r in table.values()),bounds=[dict(max_clicks=a['max_clicks'],uniform_bits=a['uniform_bits'],queries=a['queries'],tv_lower=float(a['tv_lower']),tv_upper=float(a['tv_upper']),confirmed_overlap=float(a['confirmed_overlap']),seen_failure_mass=float(a['seen_failure_mass']),target_selected_lower=float(a['target_selected_lower']),target_selected_upper=float(a['target_selected_upper'])) for a in audits],whitening_residual=float(cert.delta),maximum_certificate_atom_width=float(max(hi-lo for lo,hi in cert_atoms.values())),seconds=perf_counter()-started)
    save(out/'case_37.json',row);unchanged=hashes=={n:digest(ROOT/n) for n in names};assert unchanged;save(out/'summary.json',dict(planned=1,finished=1,completed=1,unchanged_sources=unchanged,dependencies=dependencies,original_failure_preserved=True,integer_string_digit_limit=0));print(json.dumps(row['bounds'],indent=2))
if __name__=='__main__':main()
