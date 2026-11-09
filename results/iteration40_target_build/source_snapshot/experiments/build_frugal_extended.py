"""Build a separate pinned Frugal adapter with bounded output and audit hooks."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path
from build_frugal import ROOT,UPSTREAM_COMMIT,SAMPLER_BODY_SHA256,tree_fingerprint


def instrumentation(upstream):
    source=(upstream/'include/fgbs/cpu/cpu_sampler.hpp').read_text()
    a=source.index('template<usize N, typename T>\nvoid update_result(')
    b=source.index('template<usize N, typename T>\nSampleResult<N, T> generate_sample(',a)
    c=source.index('\ntemplate<usize N>\nResult<std::vector<Bitstring<N>>> generate_samples(',b)
    d=source.index('\n#undef MARGINAL_END_ZERO_DIFF',c)
    update=source[a:b].rstrip();sample=source[b:c];batch=source[c:d].strip()
    assert hashlib.sha256(sample.encode()).hexdigest()==SAMPLER_BODY_SHA256
    hashes={name:hashlib.sha256(value.encode()).hexdigest() for name,value in (('update_result',update),('generate_sample',sample),('generate_samples',batch))}
    branch='        update_result<N>(rand, result, prev_prob, parity_arr, i, zero_next_prob);'
    assert sample.count(branch)==1
    audited_update=update.replace('void update_result(','void update_result_audited(').replace('usize i, T zero_next_prob) {','usize i, T zero_next_prob, AuditRecord* records) {')
    old='        const auto mask = rand() > zero_prob;'
    new='''        const auto random_value = rand();
        const auto mask = random_value > zero_prob;
        for (usize j = 0; j < ProbSimd::size; ++j)
            records[j*N+i] = {random_value.get(j), zero_prob.get(j), prev_prob.get(j), zero_next_prob.get(j)};'''
    assert old in audited_update;audited_update=audited_update.replace(old,new)
    audited_sample=sample.replace('generate_sample(','generate_sample_audited(',1).replace('T* FGBS_INTERNAL_RESTRICT double_elided_marginal_arr) {','T* FGBS_INTERNAL_RESTRICT double_elided_marginal_arr, AuditRecord* records) {',1).replace(branch,'        update_result_audited<N>(rand, result, prev_prob, parity_arr, i, zero_next_prob, records);')
    audited_batch=batch.replace('generate_samples(','generate_samples_audited(',1).replace('usize sample_count, uint seed) {','usize sample_count, uint seed, std::vector<AuditRecord>& records) {',1)
    audited_batch=audited_batch.replace('    #pragma omp parallel default(none) shared(samples, seeds, bitstring_layout, sampling_target) firstprivate(batch_size, tail_size, batch_count)','    records.resize(batch_count * batch_size * N);\n    #pragma omp parallel default(none) shared(samples, seeds, bitstring_layout, sampling_target, records) firstprivate(batch_size, tail_size, batch_count)')
    original='generate_sample<N, ProbSimd>(bitstring_layout, sampling_target, rand, partial_marginal_arr.data(), elided_marginal_arr.data(), partial_elided_marginal_arr.data(), double_elided_marginal_arr.data())'
    assert original in audited_batch;audited_batch=audited_batch.replace(original,original.replace('generate_sample<','generate_sample_audited<').replace('double_elided_marginal_arr.data())','double_elided_marginal_arr.data(), records.data() + i*batch_size*N)'))
    prefix=sample[sample.index(' {\n')+3:]
    prefix=prefix.replace('    SampleResult<N, T> result{};','    using T = ProbSimd;')
    prefix=prefix.replace(branch,'''        const auto threshold = zero_next_prob / prev_prob;
        if (i == forced_prefix.size()) return {threshold.get(0), prev_prob.get(0), zero_next_prob.get(0)};
        const auto mask = forced_prefix[i] == '1' ? get_one<T>() > get_zero<T>() : get_one<T>() < get_zero<T>();
        parity_arr[i] = xsimd::select(mask, -get_half<T>(), get_half<T>());
        prev_prob = xsimd::select(mask, prev_prob - zero_next_prob, zero_next_prob);''')
    prefix=prefix.replace('    return result;','    throw std::runtime_error("Prefix must be shorter than the mode count");')
    prefix='''template<usize N>
PrefixRecord trace_prefix(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target,
    std::string const& forced_prefix, ProbSimd* partial_marginal_arr, ProbSimd* elided_marginal_arr,
    ProbSimd* partial_elided_marginal_arr, ProbSimd* double_elided_marginal_arr) {
'''+prefix
    text='''// Generated audit hooks; NOT an official FrugalGBS API.
/*
'''+(upstream/'LICENSE.md').read_text()+'''\n*/
#pragma once
#include "fgbs/cpu/cpu_sampler.hpp"
#include <string>
#include <stdexcept>
#define GBSLAB39_RESTRICT __restrict__
namespace fgbs::cpu {
struct AuditRecord { double uniform, threshold, previous, proposed; };
struct PrefixRecord { double threshold, previous, proposed; };
'''+('\n'.join((audited_update,audited_sample,audited_batch,prefix))).replace('FGBS_INTERNAL_RESTRICT','GBSLAB39_RESTRICT')+'\n}\n#undef GBSLAB39_RESTRICT\n'
    return text,hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--eigen',type=Path,required=True);p.add_argument('--compiler',required=True);p.add_argument('--modes',default='2,4,10,32,64,144');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);upstream=ROOT/'third_party/FrugalGBS';xsimd=ROOT/'third_party/xsimd/include';modes=tuple(sorted(set(map(int,a.modes.split(',')))));assert modes and all(2<=n<=256 for n in modes);compiler=Path(a.compiler).resolve();assert compiler.is_file() and (a.eigen/'Eigen/Core').is_file();generated=out/'generated';(generated/'fgbs').mkdir(parents=True)
    header,body_hashes=instrumentation(upstream);(generated/'frugal_extended_hooks.hpp').write_text(header);(generated/'frugal_extended_modes.hpp').write_text('#define GBSLAB_FRUGAL_EXTENDED_MODES(M) '+' '.join(f'M({n})' for n in modes)+'\n');global_header=(upstream/'include/fgbs/global.hpp.in').read_text().replace('@FGBS_REAL_PRECISION@','2').replace('@FGBS_MAX_CUTOFF_ORDER@','5');(generated/'fgbs/global.hpp').write_text(global_header)
    names=['experiments/build_frugal_extended.py','experiments/build_frugal.py','experiments/frugal_extended.cpp',*[p.relative_to(ROOT).as_posix() for p in upstream.rglob('*') if p.is_file()],*[p.relative_to(ROOT).as_posix() for p in xsimd.rglob('*') if p.is_file()]];hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    for n in names:dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    sources=[ROOT/'experiments/frugal_extended.cpp',*sorted((upstream/'src/fgbs/core').glob('*.cpp')),upstream/'src/fgbs/layout.cpp',upstream/'src/fgbs/cpu/cpu_sampler.cpp'];exe=out/'frugal_extended.exe';cmd=[str(compiler),'-O3','-std=c++20','-fopenmp','-DFGBS_GPU_ENABLED=0','-I',str(generated),'-I',str(upstream/'include'),'-I',str(a.eigen),'-I',str(xsimd),*map(str,sources),'-o',str(exe)];meta=dict(upstream_commit=UPSTREAM_COMMIT,source_hashes=hashes,body_hashes=body_hashes,compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),compiler_version=subprocess.check_output([str(compiler),'--version'],text=True).splitlines()[0],eigen=tree_fingerprint(a.eigen),xsimd=tree_fingerprint(xsimd),command=cmd,modes=modes,precision='binary64',cutoff_max=5,gpu=False);(out/'plan.json').write_text(json.dumps(meta,indent=2)+'\n');start=time.perf_counter()
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:proc=subprocess.run(cmd,stdout=stdout,stderr=stderr)
    meta.update(returncode=proc.returncode,seconds=time.perf_counter()-start,unchanged_sources=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==sha for n,sha in hashes.items()))
    if proc.returncode==0:
        meta['executable_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest();meta['runtime_files']={}
        for dll in ('libgcc_s_seh-1.dll','libgomp-1.dll','libstdc++-6.dll','libwinpthread-1.dll'):
            source=compiler.parent/dll;shutil.copy2(source,out/dll);meta['runtime_files'][dll]=hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:meta[k] for k in ('returncode','seconds','unchanged_sources')}),flush=True)
    if proc.returncode:raise RuntimeError('Build failed; logs preserved')
if __name__=='__main__':main()
