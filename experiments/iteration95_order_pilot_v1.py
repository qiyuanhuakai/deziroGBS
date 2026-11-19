"""Paired mode-order controls using the same native kernel and truncation budget."""
import argparse
from iteration94_common_v2 import ROOT, Path, np, H, load93, sample93, save, manifest

def order_for(u, t):
    # Pair weight: overlap of latent squeezed populations at the same detector.
    intensity = np.abs(u)**2 * (t*t/(1-t*t))
    total = intensity.sum(axis=1)
    normalized = np.divide(intensity,total[:,None],out=np.zeros_like(intensity),where=total[:,None]>0)
    weights = intensity.T @ normalized
    weights = (weights+weights.T)/2
    np.fill_diagonal(weights,0.)
    eigenvalues, vectors = np.linalg.eigh(np.diag(weights.sum(axis=1))-weights)
    direction = vectors[:,1]
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    order = np.argsort(direction,kind='stable')
    return order, weights, eigenvalues

def physical_error(before, after):
    # Compare the entire Gaussian kernel in the original optical column space.
    # The permutation merely permutes the noise-factor rows; columns address
    # the SAME normal variates. This also preserves each paired displacement.
    u = before['u']
    transfer = np.linalg.lstsq(u,after['u'],rcond=None)[0]
    real = np.block([[transfer.real,-transfer.imag],[transfer.imag,transfer.real]])
    def excess(model):
        t = model['t']
        return np.diag(np.r_[t/(1-t),-t/(1+t)]) + 2*model['noise']@model['noise'].T
    return max(float(np.abs(v).max()) for v in (
        u@transfer-after['u'], real@excess(after)@real.T-excess(before),
        real@after['noise']-before['noise'],transfer@after['alpha']-before['alpha']))

parser = argparse.ArgumentParser()
parser.add_argument('--output',required=True,type=Path)
out = parser.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
cases = [('pure4',0,128,16,128,.01),('displaced4',1,128,16,128,.01),
         ('pure8',2,128,8,32,.02),('actual0',0,4,6,32,.1),('actual1',1,4,6,32,.01)]
inputs={}
for name,j,*_ in cases:
    inputs[name]=ROOT/(f'results/iteration94_actual_quality_v1/input_{j}.npz' if name.startswith('actual') else f'results/iteration93_reference_v2/input_{j}.npz')
files=[Path(__file__),ROOT/'native/gaussian_mps_core94_v2.cpp',ROOT/'results/iteration94_build_v2/mps.dll',*inputs.values()]
save(out,'plan.json',dict(scope=__doc__,cases=cases,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},
    threads=4,cap=256,formal_quality_records=0,experiment='0x95001000+small case, actual both use 0x95001100 with distinct sectors',
    design='One fixed four-worker batch per order and case. Paired random addresses across orders. Alternate execution order. Single-run elapsed times are exploratory.'))
lib,runtime=load93()
save(out,'backend.json',lib._backend)
rows=[]
for i,(name,j,n,d,bond,budget) in enumerate(cases):
    with np.load(inputs[name]) as archive:
        z={k:archive[k] for k in archive.files}
    order,weights,spectrum=order_for(z['u'],z['t'])
    m=len(order)
    alternate=dict(u=z['u'][:,order],t=z['t'][order],alpha=z['alpha'][order],noise=z['noise'][np.r_[order,m+order]])
    error=physical_error(z,alternate)
    assert error<2e-11
    def cut_weight(p):
        position=np.argsort(p)
        return float(np.sum(weights*np.abs(position[:,None]-position[None,:]))/2)
    payload=dict(order=order.tolist(),input_error=error,identity_cut=cut_weight(np.arange(m)),
                 reordered_cut=cut_weight(order),eigen_gap=float(spectrum[2]-spectrum[1]))
    save(out,name+'_order.json',payload)
    draws={}
    for label in (('original','reordered') if i%2==0 else ('reordered','original')):
        model=z if label=='original' else alternate
        actual=name.startswith('actual')
        raw,meta=sample93(lib,model['u'],model['t'],model['alpha'],n,0x95001100 if actual else 0x95001000+j,
            sector=j if actual else 0,threads=4,d=d,bond=bond,budget=budget,cap=256,noise=model['noise'])
        assert np.isin(raw['errors'],[0,2]).all()
        assert (raw['statistics'][:,5]<=budget).all()
        np.savez_compressed(out/f'{name}_{label}.npz',**raw)
        row=dict(case=name,order=label,**meta,photons=raw['statistics'][:,0].astype(int).tolist(),
                 charge=raw['statistics'][:,5].tolist(),errors=raw['errors'].tolist())
        save(out,f'{name}_{label}.json',row)
        rows.append(row);draws[label]=raw
        print(name,label,meta,flush=True)
    payload.update(equal_count_rows=int(np.all(draws['original']['counts']==draws['reordered']['counts'],axis=1).sum()),
        equal_error_flags=int(np.count_nonzero(draws['original']['errors']==draws['reordered']['errors'])),
        equal_draw_counters=int(np.all(draws['original']['statistics'][:,14:17]==draws['reordered']['statistics'][:,14:17],axis=1).sum()))
    save(out,name+'_comparison.json',payload)
save(out,'summary.json',dict(completed=True,rows=rows))
manifest(out)
