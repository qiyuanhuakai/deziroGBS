"""Test one reduced-state reset against conditional vacuum probabilities.

The bound concerns the first click following a vacuum prefix. No iid samples
or runtime claim for a full sampler are produced by this diagnostic.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--source', type=Path, default=ROOT/'native/gaussian_reduced_click95_v1.cpp')
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, data):
    (out/name).write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')

def optical_input(path):
    with np.load(path) as model:
        u, t, alpha, noise = (model[key] for key in ('u','t','alpha','noise'))
    real = np.block([[u.real,-u.imag],[u.imag,u.real]])
    core = np.diag(np.r_[t/(1-t),-t/(1+t)]) + 2*noise@noise.T
    q = np.eye(2*len(u)) + real@core@real.T
    amplitude = u@alpha
    # Weyl bound for I+2 R core R^T; valid also for a slightly imperfect isometry.
    negative_eigenvalue = min(0., float(np.linalg.eigvalsh(core).min()))
    minimum = 1 + 2 * np.linalg.norm(u,2)**2 * negative_eigenvalue
    assert minimum > 1e-8
    return q, np.sqrt(2)*np.r_[amplitude.real, amplitude.imag], float(minimum-1e-9)

def logvac(q, mean, ports):
    if not ports:
        return 0.
    indices = list(ports) + [p+len(mean)//2 for p in ports]
    lower = np.linalg.cholesky(q[np.ix_(indices,indices)])
    white = np.linalg.solve(lower,mean[indices])
    return -np.log(lower.diagonal()).sum() - (white @ white)/2

def vacuum_tables_to_law(vac):
    # Reverse subset transform: vacuum-set probabilities -> exact no-click sets.
    masses = vac.copy()
    for bit in range((len(vac)-1).bit_length()):
        for mask in range(len(vac)):
            if not mask & (1<<bit):
                masses[mask] -= masses[mask | (1<<bit)]
    return masses

compiler = Path(shutil.which('g++'))
source = args.source.resolve()
dll = out/'reduced.dll'
small_paths = [ROOT/f'results/iteration93_reference_v2/input_{i}.npz' for i in range(4)]
actual_paths = [ROOT/f'results/iteration94_actual_quality_v1/input_{i}.npz' for i in range(2)]
history_path = ROOT/'results/iteration94_branch_actual_pilot_v1/raw_0_8.npz'
with np.load(history_path) as history:
    first_ports = sorted({int(np.flatnonzero(row)[0]) for row in history['clicks']})
save('plan.json', dict(scope=__doc__,actual_first_ports=first_ports,selected_before_probe=True,
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in [*small_paths,*actual_paths,history_path]},
    driver_sha256=digest(Path(__file__)),cpp_sha256=digest(source)))
command = [str(compiler),'-O3','-std=c++20','-shared','-fno-fast-math','-ffp-contract=off',
    '-DEIGEN_DONT_PARALLELIZE','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),
    str(source),'-o',str(dll)]
build = subprocess.run(command, capture_output=True,text=True)
save('build.json',dict(command=command,returncode=build.returncode,stderr=build.stderr))
build.check_returncode()
runtime = os.add_dll_directory(str(compiler.parent))
probe = ctypes.CDLL(str(dll)).reduced95_probe
array = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
probe.argtypes = [ctypes.c_int,ctypes.c_int,array,array,ctypes.c_double,array,array]
rows = []

def evaluate(case, q, mean, minimum, ports):
    q, mean = np.ascontiguousarray(q), np.ascontiguousarray(mean)
    modes = len(mean)//2
    for clicked in ports:
        info, future = np.zeros(6), np.zeros((modes,3))
        rc = probe(modes,clicked,q,mean,minimum,info,future)
        assert rc==0,(case,clicked,rc)
        prefix = list(range(clicked))
        base = logvac(q,mean,prefix)
        p1 = -np.expm1(logvac(q,mean,prefix+[clicked])-base)
        assert abs(info[1]-p1)<2e-12 and abs(info[0]-np.exp(base))<2e-12
        suffix = list(range(clicked+1,modes))
        def projected(subset):
            unconditioned = np.exp(logvac(q,mean,prefix+subset)-base)
            joint_vac = np.exp(logvac(q,mean,prefix+[clicked]+subset)-base)
            return unconditioned, (unconditioned-joint_vac)/p1
        checkpoints = sorted(set(suffix if modes<=8 else suffix[::83]+suffix[-1:]+[int(info[5])]) & set(suffix))
        for j in checkpoints:
            assert np.max(np.abs(future[j,:2]-projected([j])))<5e-10
        row = dict(case=case,port=clicked,prefix_probability=info[0],conditional_click_probability=info[1],
            history_probability=info[0]*info[1],single_port_tv_lower=info[2],trace_distance_upper=info[3],
            shift_energy=info[4],strongest_port=int(info[5]),checked_marginals=len(checkpoints))
        if modes<=8:
            vac = np.array([projected([port for bit,port in enumerate(suffix) if mask & (1<<bit)])
                for mask in range(1<<len(suffix))])
            reset_law, exact_law = map(vacuum_tables_to_law,vac.T)
            assert min(reset_law.min(),exact_law.min())>-2e-11
            distance = np.abs(exact_law-reset_law).sum()/2
            assert info[2]-1e-11<=distance<=info[3]+1e-11
            row['complete_future_click_tv']=float(distance)
            if case=='product':
                assert info[3]==0 and distance<1e-12
            if case=='tmsv' and clicked==0:
                assert max(abs(distance-1/2.3),abs(info[3]-1/2.3))<1e-12
        np.savez_compressed(out/f'{case}_{clicked}.npz',future=future,summary=info)
        rows.append(row)

thermal = np.array([.2,.7,1.3])
evaluate('product',np.diag(1+np.r_[thermal,thermal]),np.array([.3,.2,.5,-.1,.4,.1]),1.,[0,1])
n = 1.3
c = np.sqrt(n*(1+n))
q = np.eye(4)*(1+n)
q[0,1]=q[1,0]=c
q[2,3]=q[3,2]=-c
evaluate('tmsv',q,np.zeros(4),1+2*n-2*c,[0])
for case,path in zip(('pure4','displaced4','pure8','lossy'),small_paths):
    q, mean, minimum = optical_input(path)
    evaluate(case,q,mean,minimum,[0,1])
for sector,path in enumerate(actual_paths):
    q, mean, minimum = optical_input(path)
    evaluate('actual'+str(sector),q,mean,minimum,first_ports)
save('validation.json',dict(passed=True,rows=rows,complete_future_laws=sum('complete_future_click_tv' in r for r in rows)))
save('manifest.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
print(json.dumps(rows,indent=2),flush=True)
