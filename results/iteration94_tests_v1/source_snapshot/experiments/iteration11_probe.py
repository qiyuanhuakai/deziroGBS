"""Frozen-input MPS feasibility and small-system construction/measurement probe."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gbslab.gaussian import haar_unitary, local_unitary, covariance_from_interferometer
from gbslab.exact import ThresholdModel
from gbslab.mps_probe import (schmidt_parameters, geometric_top_weights,
    dense_to_mps, mps_to_dense, fixed_threshold_law, sample_threshold)
from iteration10_reference import (core_parameters, core_state_reference,
    fixed_displacement_law, gaussian_averaged_law)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def spectral_order(u):
    weights = np.abs(u @ u.T)**2
    np.fill_diagonal(weights, 0)
    _, vectors = np.linalg.eigh(np.diag(weights.sum(axis=1))-weights)
    f = vectors[:, 1]
    if f[np.argmax(np.abs(f))] < 0:
        f = -f
    return np.argsort(f, kind='stable')


def spectrum_panel(cfg, output):
    rows = []
    for m in cfg['spectral_modes']:
        for seed in cfg['seeds']:
            for ensemble in cfg['ensembles']:
                rng = np.random.default_rng(seed)
                u = haar_unitary(m, rng) if ensemble == 'haar' else local_unitary(m, 4, rng)
                for profile in cfg['profiles']:
                    label = f"m{m}_{ensemble}_{profile['name']}_s{seed}"
                    s = core_parameters(profile['squeezing'], profile['transmissivity'])['core_squeezing']
                    v = covariance_from_interferometer(u, s)
                    np.savez_compressed(output / (label+'.npz'), U=u, pure_covariance=v)
                    for ordering in cfg['orders']:
                        order = np.arange(m) if ordering == 'natural' else spectral_order(u)
                        cuts = list(range(1, m)) if m <= 16 else [m//4, m//2, 3*m//4]
                        for cut in cuts:
                            start = perf_counter()
                            row = {'input': label, 'modes': m, 'ensemble': ensemble,
                                'profile': profile['name'], 'seed': seed, 'ordering': ordering,
                                'order': order.tolist(), 'cut': cut}
                            try:
                                q = schmidt_parameters(v, order[:cut])
                                p = geometric_top_weights(q, cfg['rank_cap'])
                                cumulative = np.cumsum(p, dtype=np.longdouble)
                                targets = cfg['cut_tail_targets']
                                ranks = {}
                                for target in targets:
                                    ix = np.flatnonzero(cumulative >= 1-target)
                                    ranks[str(target)] = int(ix[0]+1) if len(ix) else None
                                row.update(status='complete', geometric_parameters=q.tolist(),
                                    required_rank=ranks, rank_cap=cfg['rank_cap'],
                                    tail_at_cap=float(max(0., 1-cumulative[-1])),
                                    seconds=perf_counter()-start)
                            except Exception as e:
                                row.update(status='failed', error=repr(e), seconds=perf_counter()-start)
                            rows.append(row)
                    save(output / 'spectral_rows.json', rows)
                    print('spectral', label, flush=True)
    return rows


def rectangular_state(x):
    occ = np.indices(x.shape).reshape(x.ndim, -1).T
    return {'occupations': occ, 'amplitudes': x.ravel(), 'cutoff': sum(d-1 for d in x.shape)}


def compression_panel(cfg, output):
    rows = []
    for m in cfg['compression_modes']:
        for seed in cfg['seeds']:
            u = haar_unitary(m, np.random.default_rng(seed))
            for profile in cfg['profiles']:
                if profile['name'] not in cfg['compression_profiles']:
                    continue
                for cutoff in cfg['cutoffs']:
                    label = f"m{m}_{profile['name']}_s{seed}_K{cutoff}"
                    start = perf_counter()
                    state = core_state_reference(u, profile['squeezing'], profile['transmissivity'], cutoff)
                    x = np.zeros((cutoff+1,)*m, dtype=complex)
                    x[tuple(state['occupations'].T)] = state['amplitudes']
                    prepare_seconds = perf_counter()-start
                    normal = np.random.default_rng(seed+1000).normal(size=m)
                    displace = state['displacement_real_factor'] @ normal
                    alpha = (displace[:m]+1j*displace[m:])/2
                    full_law = fixed_displacement_law(state, alpha)['distribution']
                    np.savez_compressed(output/(label+'_input.npz'), U=u, tensor=x,
                        alpha=alpha, reference_fixed_law=full_law,
                        displacement_factor=state['displacement_real_factor'])
                    averaged = m == 4 and cutoff == 2
                    if averaged:
                        target = ThresholdModel(state['target_covariance']).full_distribution()
                        core_law = gaussian_averaged_law(state)['distribution']
                    for bond in cfg['bonds']:
                        row = {'input': label, 'modes': m, 'profile': profile['name'],
                            'seed': seed, 'cutoff': cutoff, 'max_bond': bond,
                            'core_prepare_seconds': prepare_seconds,
                            'core_tail': state['tail_probability'],
                            'core_tail_trace_bound': state['trace_distance_bound']}
                        start = perf_counter()
                        try:
                            tensors, diagnostics = dense_to_mps(x, bond)
                            row['compression_seconds'] = perf_counter()-start
                            row.update(diagnostics)
                            law = fixed_threshold_law(tensors, alpha)
                            row['fixed_displacement_tv_vs_core'] = float(.5*np.abs(law-full_law).sum())
                            row['contraction_check'] = row['fixed_displacement_tv_vs_core'] <= diagnostics['state_trace_distance']+1e-10
                            np.savez_compressed(output/(label+f'_D{bond}.npz'),
                                **{f'A{j}': a for j, a in enumerate(tensors)}, fixed_law=law)
                            # Full correlated displacement per draw; no reuse of one fixed alpha.
                            rng = np.random.default_rng(seed+2000)
                            noise = rng.normal(size=(cfg['samples'], m)) @ state['displacement_real_factor'].T
                            alphas = (noise[:, :m]+1j*noise[:, m:])/2
                            timing = perf_counter()
                            samples = sample_threshold(tensors, alphas, rng)
                            row['sampling_seconds'] = perf_counter()-timing
                            row['samples'] = len(samples)
                            row['total_prepare_and_sampling_seconds'] = prepare_seconds+row['compression_seconds']+row['sampling_seconds']
                            np.savez_compressed(output/(label+f'_D{bond}_samples.npz'), samples=samples, alpha=alphas)
                            row['status'] = 'complete'
                            if averaged:
                                try:
                                    comp_state = rectangular_state(mps_to_dense(tensors))
                                    comp_state['displacement_covariance'] = state['displacement_covariance']
                                    result = gaussian_averaged_law(comp_state)
                                    q = result['distribution']
                                    row['averaged_tv_vs_target'] = float(.5*np.abs(q-target).sum())
                                    row['averaged_tv_vs_core'] = float(.5*np.abs(q-core_law).sum())
                                    row['uncompressed_averaged_tv_vs_target'] = float(.5*np.abs(core_law-target).sum())
                                    row['construction_trace_budget'] = min(1., state['trace_distance_bound']+diagnostics['state_trace_distance'])
                                    row['averaged_status'] = 'complete'
                                    row['averaged_contraction_check'] = row['averaged_tv_vs_core'] <= diagnostics['state_trace_distance']+1e-10
                                    np.savez_compressed(output/(label+f'_D{bond}_averaged.npz'), target=target, core=core_law, compressed=q, raw_compressed=result['raw_distribution'])
                                    row['averaged_reference_diagnostics'] = result['diagnostics']
                                except Exception as e:
                                    row.update(averaged_status='failed', averaged_error=repr(e))
                        except Exception as e:
                            row.update(status='failed', error=repr(e))
                        row['entire_row_seconds'] = perf_counter()-start
                        rows.append(row)
                        save(output/'compression_rows.json', rows)
                        print('compression', label, bond, row['status'], row.get('averaged_status', ''), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/iteration11_mps_probe.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    save(args.output/'plan.json', cfg)
    files = ['src/gbslab/mps_probe.py', 'src/gbslab/gaussian.py', 'src/gbslab/exact.py',
        'experiments/iteration11_probe.py', 'experiments/iteration10_reference.py',
        'tests/test_mps_probe.py', 'configs/iteration11_mps_probe.json']
    hashes = {}
    for f in files:
        dst = args.output/'source_snapshot'/f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/f, dst)
        hashes[f] = hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    save(args.output/'metadata.json', {'date':'2026-11-08', 'python':sys.version,
        'numpy':np.__version__, 'scipy':scipy.__version__, 'platform':platform.platform(),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256':hashes, 'scope':cfg['scope']})
    s = spectrum_panel(cfg, args.output)
    c = compression_panel(cfg, args.output)
    save(args.output/'completion.json', {'spectral_rows':len(s), 'compression_rows':len(c),
        'spectral_failures':sum(r['status']!='complete' for r in s),
        'compression_failures':sum(r['status']!='complete' for r in c),
        'averaged_complete':sum(r.get('averaged_status')=='complete' for r in c),
        'averaged_failures':sum(r.get('averaged_status')=='failed' for r in c)})


if __name__ == '__main__':
    main()
