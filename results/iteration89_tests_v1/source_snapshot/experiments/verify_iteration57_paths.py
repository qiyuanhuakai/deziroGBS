"""Independent provenance, complete-path arithmetic and uint64 word audit."""
import sys,json,gzip,hashlib,shutil,math
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import cdf_count,replay
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration57_paths_validation';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration57_paths','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];count=0;shards=[];total_steps=total_paths=public_steps=0
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);b=(ROOT/n).read_bytes();return decode(json.loads(gzip.decompress(b) if n.endswith('.gz') else b))
    def flush():
        if checks:
            name=f'checks_{len(shards):03d}.json';save(out/name,checks);shards.append(name);checks.clear()
    def check(n,ok):
        nonlocal count
        count+=1;checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
        if len(checks)>=20000:flush()
    build=read('results/iteration57_build/build.json');oldbuild=read('results/iteration39_build/build.json');oldmanifest=read('results/iteration39_manifest.json');check('old build bound',deps['results/iteration39_build/build.json']==oldmanifest['artifacts']['results/iteration39_build/build.json'])
    check('successful pinned build',build['returncode']==0 and build['unchanged_sources'] and build['compiler_sha256']==oldbuild['compiler_sha256']==digest(build['command'][0]))
    for n,s in build['source_hashes'].items():check('build source '+n,digest(ROOT/n)==s==digest(ROOT/'results/iteration57_build/source_snapshot'/n))
    for n,s in oldbuild['source_hashes'].items():check('unchanged upstream '+n,build['source_hashes'][n]==s)
    for n,s in build['dependencies'].items():check('build dependency '+n,digest(ROOT/n)==s);deps[n]=s
    command=oldbuild['command'].copy();command[command.index(str(ROOT/'experiments/frugal_extended.cpp'))]=str(ROOT/'experiments/iteration57_paths.cpp');command[-1]=str(ROOT/'results/iteration57_build/frugal_paths.exe');check('only adapter and output command changes',command==build['command']);deps['results/iteration57_build/frugal_paths.exe']=digest(ROOT/'results/iteration57_build/frugal_paths.exe');check('executable bound',deps['results/iteration57_build/frugal_paths.exe']==build['executable_sha256'])
    for n,s in build['runtime_files'].items():p=ROOT/'results/iteration57_build'/n;check('runtime '+n,digest(p)==s==oldbuild['runtime_files'][n]);deps[p.relative_to(ROOT).as_posix()]=s
    original=(ROOT/'results/iteration39_build/generated/frugal_extended_hooks.hpp').read_text();a=original.index('template<usize N>\nPrefixRecord trace_prefix(');b=original.index('\n}\n',a)+3;body=original[a:b]
    body=body.replace('PrefixRecord trace_prefix(', 'std::vector<PrefixRecord> trace_path(').replace('    using T = ProbSimd;','    using T = ProbSimd;\n    if (forced_prefix.size() != N || forced_prefix.find_first_not_of("01") != std::string::npos) throw std::runtime_error("Complete binary path required");\n    std::vector<PrefixRecord> records;').replace('if (i == forced_prefix.size()) return {threshold.get(0), prev_prob.get(0), zero_next_prob.get(0)};','records.push_back({threshold.get(0), prev_prob.get(0), zero_next_prob.get(0)});').replace('throw std::runtime_error("Prefix must be shorter than the mode count");','return records;')
    check('exact limited hook transformation',(ROOT/'experiments/iteration57_path_hooks.hpp').read_text()=='#pragma once\n#include "frugal_extended_hooks.hpp"\nnamespace fgbs::cpu {\n'+body+'\n}\n')
    plan=read('results/iteration57_paths/plan.json');summary=read('results/iteration57_paths/summary.json');check('five completed jobs',summary['planned']==summary['finished']==summary['completed']==5 and summary['unchanged_sources'] and summary['unchanged_dependencies'])
    for n,s in summary['source_hashes'].items():check('producer source '+n,digest(ROOT/n)==s==digest(ROOT/'results/iteration57_paths/source_snapshot'/n))
    for n,s in summary['dependencies'].items():check('producer dependency '+n,digest(ROOT/n)==s);deps[n]=s
    for job in plan['jobs']:
        label=job['label'];n=job['modes'];tag=label+' ';folder='results/iteration57_paths/';meta=read(folder+label+'.storage.json');rawbytes=(ROOT/folder/(label+'.stdout.json.gz')).read_bytes();payload=gzip.decompress(rawbytes);check(tag+'gzip bytes',len(rawbytes)==meta['compressed_bytes'] and len(payload)==meta['raw_bytes'] and hashlib.sha256(payload).hexdigest()==meta['raw_sha256'] and rawbytes[4:8]==bytes(4));raw=read(folder+label+'.stdout.json.gz');audit=read(folder+label+'.audit.json');old=read(job['reference']);check(tag+'identity',raw['modes']==n and raw['cutoff']==job['cutoff'] and raw['samples']==job['samples'] and raw['seed']==job['seed'] and raw['prefix_depth']==job['depth'] and audit['job']==job)
        check(tag+'unchanged public paths',raw['outputs']==old['outputs'] and raw['audit_paths']==old['audit_paths']);steps=replay({**raw,'prefix_tree':[]});public_steps+=steps;check(tag+'independent public replay',steps==audit['replayed_steps']==raw['public_path_checks']==n*job['samples'])
        table={r['prefix']:r for r in old['prefix_tree']};check(tag+'old complete control prefix',len(raw['prefix_tree'])==(1<<job['depth'])-1 and all(table[r['prefix']]==r for r in raw['prefix_tree']) and raw['prefix_reverse_verified'])
        outputs=[''.join(str(mask>>j&1) for j in range(10)) for mask in range(1024)] if n==10 else ['0'*144]+[''.join('1' if j==pos else '0' for j in range(144)) for pos in (0,47,95,143)]
        check(tag+'prescribed outputs',[r['output'] for r in raw['path_probes']]==outputs==[r['output'] for r in audit['laws']]);check(tag+'native original-prefix and reverse controls',raw['probe_control_checks']==len(outputs)*(1 if n==10 else 5) and raw['probe_reverse_verified']);shared={};probabilities={'continuous':[],'uint64':[]}
        for path_index,(probe,record) in enumerate(zip(raw['path_probes'],audit['laws'])):
            output=probe['output'];check(tag+f'path{path_index} length',len(probe['steps'])==n);previous=1.;masses={k:F(1) for k in probabilities};factors={k:[] for k in probabilities};total_paths+=1
            for j,(bit,step) in enumerate(zip(output,probe['steps'])):
                q,prev,z=map(float.fromhex,step);total_steps+=1
                correct=math.isnan(q) if prev==0 and z==0 else (math.isinf(q) and q>0) if prev==0 else q==float(F(z)/F(prev))
                check(tag+f'path{path_index} step{j} recurrence',prev==previous and correct and math.isfinite(prev) and math.isfinite(z))
                previous=float(F(prev)-F(z)) if bit=='1' else z
                if j<(10 if n==10 else 16):check(tag+f'path{path_index} step{j} old tree',step==[table[output[:j]][key] for key in ('threshold_hex','previous_hex','proposed_hex')])
                if output[:j] in shared:check(tag+f'path{path_index} step{j} shared prefix',shared[output[:j]]==step)
                else:shared[output[:j]]=step
                for model in probabilities:
                    zero=F(cdf_count(q),1<<64) if model=='uint64' else F(1) if math.isnan(q) or q>=1 else F(0) if q<0 else F(q);factor=zero if bit=='0' else 1-zero;masses[model]*=factor;factors[model].append(factor)
            for model in probabilities:
                check(tag+f'path{path_index} {model} exact product',record['models'][model]['probability']==masses[model] and record['models'][model]['factors']==factors[model] and record['models'][model]['model']==model);probabilities[model].append(masses[model])
        if n==10:
            legacy=read('results/iteration39_local/case_03.audit.json')
            for model in probabilities:check(tag+'complete law equality '+model,probabilities[model]==legacy['laws'][model]['distribution'] and sum(probabilities[model])==1)
        else:
            for model in probabilities:check(tag+'selected subprobability '+model,0<=sum(probabilities[model])<=1)
        print('verified paths',label,flush=True)
    check('coverage',total_paths==1044 and total_steps==13120 and public_steps==12362);check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));flush();save(out/'validation.json',dict(count=count,passed=count,check_shards=shards,complete_paths=total_paths,path_steps=total_steps,public_steps=public_steps,local_full_atoms=1024,source_hashes=hashes,dependencies=deps));print('path checks',count,flush=True)


if __name__=='__main__':main()
