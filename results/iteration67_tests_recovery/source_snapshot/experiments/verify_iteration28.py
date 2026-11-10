"""Verify final saved states, independent spectra and unrepaired failure ledgers."""
import argparse
import json
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import shutil
import sys
import numpy as np
import scipy.linalg as la
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian_resource import omega,pure_pairing
from gbslab.banded_gaussian_mps import pairing_covariance,pairing_distance
from gbslab.mixed_low_clicks import mixed_gaussian_low_clicks
from iteration23_support import save,digest


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--external',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False); src=Path(__file__); sha=digest(src); shutil.copy2(src,out/src.name)
    def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
    checks=[]
    def check(name,value): checks.append(dict(name=name,passed=bool(value)))
    panel=ROOT/'results/iteration28_panel'; plan=read(panel/'plan.json'); rows=read(panel/'rows.json'); summary=read(panel/'summary.json')
    check('formal completion',summary['planned']==summary['finished']==summary['completed']==len(rows)==16)
    check('one rejected margin retained',summary['accepted']==15 and not rows[15]['accepted'] and rows[14]['primary'] and rows[14]['accepted'])
    for name,value in plan['source_hashes'].items(): check('formal source '+name,digest(ROOT/name)==digest(panel/'source_snapshot'/name)==value)
    for name,value in plan['dependencies'].items(): check('dependency '+name,digest(ROOT/name)==value)
    for record in read(ROOT/'results/iteration27_source/selected.json'):
        check('external '+record['local_name'],digest(args.external/record['local_name'])==record['sha256'])
    M=np.load(args.external/'jiuzhang2_mixed_cov.npy',allow_pickle=False); author=2*np.load(args.external/'best_opt_cov_jiuzhang2_mixed_cov.npy',allow_pickle=False)
    support=np.load(panel/'active_support.npz',allow_pickle=False); O=support['basis']; W=support['active_covariance']; m=len(M)//2; k=len(W)//2
    check('50 active modes',k==50)
    check('passive isometry',np.max(abs(O.T@O-np.eye(2*k)))<1e-12 and np.max(abs(O.T@omega(m)@O-omega(k)))<1e-12)
    check('compression full input error',np.linalg.norm(np.eye(2*m)+O@(W-np.eye(2*k))@O.T-M,ord=2)<1e-10)
    check('active strict physicality',np.linalg.eigvalsh(W+1j*omega(k))[0]>.3)
    extra=[]
    for i,row in enumerate(rows):
        data=np.load(panel/f'case_{i:02d}.npz',allow_pickle=False); P=data['pure_raw']; F=data['dual_factor']; V=data['input'] if row['panel']=='small' else W; n=len(P)
        check(f'{i} dual factor',np.max(abs(F@F.T-data['dual']))<1e-12)
        check(f'{i} raw remainder retained',np.max(abs(V-P-data['remainder_raw']))<1e-13)
        check(f'{i} pure identity',np.max(abs(P@omega(n//2)@P-omega(n//2)))<1e-10)
        if row['accepted']:
            B=data['pairing']; canonical=pairing_covariance(B); L=data['noise_factor']; residual=V-canonical
            check(f'{i} saved encoded state',np.max(abs(canonical-data['pure']))<1e-13)
            minimum=float(la.eigh(residual,subset_by_index=(0,0),driver='evr')[0][0])
            check(f'{i} final positive noise',minimum>=row['margin']/4)
            check(f'{i} final noise factor',np.max(abs(L@L.T-residual))<1e-12)
        else: check(f'{i} no invalid noise factor','noise_factor' not in data.files)
        if row['panel']=='small':
            X=data['sdp_pure']; check(f'{i} independent SDP physical',np.linalg.eigvalsh(X+1j*omega(n//2))[0]>-2e-8)
            check(f'{i} independent SDP remainder',np.linalg.eigvalsh(V-X)[0]>-2e-8)
            check(f'{i} independent objective comparison',-2e-7<np.trace(P-X)<2e-4 and np.trace(X)-row['diagnostics']['dual_trace']>-2e-7)
        elif row['accepted']:
            B=data['full_pairing']; P=pairing_covariance(B); L=data['full_noise_factor']; F=data['full_dual_factor']; S=F@F.T; A=np.eye(2*m)+S
            check(f'{i} original input reconstruction',np.linalg.norm(P+L@L.T-M,ord=2)<1e-10)
            # Direct nonsymmetric symplectic eigenvalues avoid the oracle's square-root route.
            spectrum=np.linalg.eigvals(1j*omega(m)@A); phi=float(np.sum(abs(spectrum.real)))
            dual=phi-float(np.sum(S*M.T)); check(f'{i} independent full dual',abs(dual-row['full_original_diagnostics']['dual_trace'])<2e-9 and np.max(abs(spectrum.imag))<1e-10)
            low=mixed_gaussian_low_clicks(P); error=float(np.max(abs(low['probabilities']-data['low_click_probabilities'])))
            check(f'{i} all pure low clicks via covariance',error<2e-12)
            extra.append(dict(sequence=i,independent_full_dual=dual,low_click_max_absolute=error))
    B_author,canonical_author,author_roundtrip=pure_pairing(author)
    primary=np.load(panel/'case_14.npz',allow_pickle=False); primary_cov=pairing_covariance(primary['full_pairing'])
    comparison=dict(author_saved_pure_mean_photons=float((np.trace(author)-2*m)/4),author_encoding_roundtrip=author_roundtrip,
        author_vs_primary_covariance_max=float(np.max(abs(author-primary_cov))),author_vs_primary_covariance_frobenius=float(np.linalg.norm(author-primary_cov)),author_encoded_vs_primary_pure_distance=pairing_distance(B_author,primary['full_pairing']))
    save(out/'author_comparison.json',comparison)
    # Frozen pilot and failed codecs are checked against their own historical sources.
    for folder in ('iteration28_pilot','iteration28_codec','iteration28_codec_retry','iteration28_codec_guarded'):
        base=ROOT/'results'/folder; oldplan=read(base/'plan.json')
        for name,value in oldplan['source_hashes'].items(): check(folder+' frozen '+name,digest(base/'source_snapshot'/name)==value)
    for folder in ('iteration28_codec','iteration28_codec_retry'):
        old=read(ROOT/'results'/folder/'summary.json'); check(folder+' 21 failures retained',old['count']==300 and old['passed']==279)
    codec=read(ROOT/'results/iteration28_codec_guarded/summary.json'); check('guarded codec all checks',codec['count']==codec['passed']==301 and codec['host_control_word_before']==codec['host_control_word_after']==639 and not codec['forest_abi_enabled'])
    check('guarded codec binary hash',digest(ROOT/'results/iteration28_codec_guarded/scalar_codec.dll')==codec['library_sha256'])
    for item in read(ROOT/'results/iteration28_codec_guarded/checks.json'):
        check('codec '+item['kind']+' '+str(item.get('precision'))+' '+str(item.get('case',item.get('input',''))),item['passed'])
    check('verifier unchanged',digest(src)==sha)
    result=dict(count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks,verifier_sha256=sha,dependencies=plan['dependencies'],large_independent_checks=extra,author_comparison=comparison)
    save(out/'validation.json',result); print(json.dumps({k:v for k,v in result.items() if k not in ('checks','dependencies')})); assert all(c['passed'] for c in checks)


if __name__=='__main__': main()
