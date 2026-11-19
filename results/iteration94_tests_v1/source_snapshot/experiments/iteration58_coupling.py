"""Retrospective two-drive bounds and predeclared rank4 channel-only holdouts."""
import sys,json,shutil,time,itertools,math
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.gaussian_fock_bridge import physical_pairing
from gbslab.joint_displacement_coupling import quantization_moments,joint_coupling
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration58_coupling';out.mkdir(exist_ok=False);start=time.perf_counter()
    names=['research/iteration58_coupling_plan.md',*[f'experiments/{n}.py' for n in ('iteration58_coupling','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={};axes={};rows=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    manifest=read('results/iteration55_manifest.json');validation=read('results/iteration55_noise_validation/validation.json');assert validation['count']==validation['passed'] and deps['results/iteration55_noise_validation/validation.json']==manifest['artifacts']['results/iteration55_noise_validation/validation.json']
    def axis(s,N):
        key=(F(s),N,{4:2,16:3}[N])
        if key not in axes:axes[key]=quantization_moments(*key)
        return dict(sigma=key[0],cells=N,radius=key[2])
    for i in range(24):
        name=f'results/iteration55_noise/case_{i:02d}.json';r=read(name);assert deps[name]==manifest['artifacts'][name]==validation['dependencies'][name];job=r['job'];aa=[axis(g['sigma'],g['cells']) for g in r['grids']]
        for a,g in zip(aa,r['grids']):assert encode(quantization_moments(a['sigma'],a['cells'],a['radius'])['grid'])==encode(g)
        bound=joint_coupling(r['target']['generator_gram'],aa);assert bound['old_axis_upper']==r['coupling']['upper'];other=r['bridge']['full_trace_interval'][1]+r['selector_TV'][1]+r['weighted_arithmetic_TV_upper'];total=min(F(1),other+bound['best_upper'])
        assert r['full_Gaussian_TV'][1]<=total<=r['total_TV_upper'] and r['ideal_grid_TV'][1]<=bound['best_upper']
        rows.append(dict(case=i,job=job,nodes=math.prod(g['cells']+2 for g in r['grids']),bound=bound,other_errors_upper=other,old_total_upper=r['total_TV_upper'],new_total_upper=total,actual_TV=r['full_Gaussian_TV'],ideal_grid_TV=r['ideal_grid_TV']))
    save(out/'retrospective_rows.json',encode(rows))
    directions=[[(1,0),(0,0)],[(0,1),(0,0)],[(0,0),(1,0)],[(0,0),(0,1)]];holdouts=[];rank4=[]
    specifications=[dict(family='real_pair',pairing=F(1,3),phase=(1,0),rotated=False,sigmas=[F(1,8),F(3,16),F(1,4),F(1,8)]),dict(family='complex_rotated',pairing=F(2,5),phase=(0,1),rotated=True,sigmas=[F(3,16),F(1,8),F(1,4),F(3,16)])]
    for fi,spec in enumerate(specifications):
        shape=[[F(24,25),-F(7,25)],[-F(7,25),-F(24,25)]] if spec['rotated'] else [[F(0),F(1)],[F(1),F(0)]];br=[[x*spec['pairing']*spec['phase'][0] for x in row] for row in shape];bi=[[x*spec['pairing']*spec['phase'][1] for x in row] for row in shape];physical=physical_pairing(br,bi);V=physical['covariance'];w=[[F(a) for a,b in v]+[F(b) for a,b in v] for v in directions];u=[v[2:]+[-x for x in v[:2]] for v in w];gram=[[sum(u[a][i]*V[i][j]*u[b][j] for i in range(4) for j in range(4)) for b in range(4)] for a in range(4)];noise=[[4*sum(spec['sigmas'][a]**2*w[a][i]*w[a][j] for a in range(4)) for j in range(4)] for i in range(4)]
        holdouts.append(dict(**spec,pairing_real=br,pairing_imag=bi,physical=physical,directions=directions,generator_gram=gram,noise_covariance=noise,mixed_covariance=[[V[i][j]+noise[i][j] for j in range(4)] for i in range(4)]))
        for index,cells in enumerate(itertools.product((4,16),repeat=4)):
            aa=[axis(s,N) for s,N in zip(spec['sigmas'],cells)];bound=joint_coupling(gram,aa);rank4.append(dict(family=spec['family'],index=index,cells=cells,nodes=math.prod(N+2 for N in cells),bound=bound))
    save(out/'rank4_models.json',encode(holdouts));save(out/'rank4_rows.json',encode(rank4))
    for i,(key,record) in enumerate(sorted(axes.items())):save(out/f'axis_{i:02d}.json',encode(record))
    save(out/'axis_index.json',encode([dict(index=i,sigma=k[0],cells=k[1],radius=k[2]) for i,k in enumerate(sorted(axes))]))
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,retrospective_cases=24,rank4_models=2,rank4_cases=32,unique_axes=len(axes),seconds=time.perf_counter()-start,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True));print('coupling production complete',len(axes),'axes',flush=True)


if __name__=='__main__':main()
