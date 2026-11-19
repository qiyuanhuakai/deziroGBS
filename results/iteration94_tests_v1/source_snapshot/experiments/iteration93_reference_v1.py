"""Frozen complete four-mode PNR laws, complete click laws and total-count laws.
Cross-check long-double Fock recurrence with 100-digit labelled pair/singleton
enumeration, independent input-mode count convolution and vacuum subsets.
"""
import argparse,functools
from iteration93_reference_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));dll=ROOT/'results/iteration93_reference_build_v1/reference.dll';lib=C.CDLL(str(dll));files=[Path(__file__),ROOT/'experiments/iteration93_reference_common_v1.py',ROOT/'native/gaussian_fock_reference93_v1.cpp',dll];save(out,'plan.json',dict(scope=__doc__,cases=['pure4','displaced4','pure8','lossy_tmsv'],cutoff=64,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}));rows=[];rng=np.random.default_rng(930070)
for j in range(4):
    q=inputs93(j);np.savez_compressed(out/f'input_{j}.npz',**q);u,t,alpha,noise=[q[k] for k in ('u','t','alpha','noise')];m=len(u);mp.mp.dps=80;click80,_=click_reference(u,t,alpha,noise);mp.mp.dps=100;click,vac=click_reference(u,t,alpha,noise);precision=max(abs(a-b) for a,b in zip(click80,click));assert precision<mp.mpf('1e-65');row=dict(case=j,modes=m,click_precision_difference=str(precision));data=dict(click=np.array(list(map(float,click))),vacuum=np.array(list(map(float,vac))))
    if j<3:
        total,tail,z,tailbound=total_reference(t,alpha,64);data.update(total=np.array(list(map(float,total))),total_tail=np.array(float(tail)));row.update(total_tail=str(tail),tail_chernoff=str(tailbound),chernoff_z=str(z));assert tail<=tailbound
    if j<2:
        keys,probs,stats,B,linear,lognorm=fock_reference(lib,u,t,alpha,64);data.update(keys=keys,probabilities=probs);occup=np.array([(keys>>(8*i))&255 for i in range(m)]).T;totals=occup.sum(1).astype(int);marginal=np.bincount(totals,weights=probs,minlength=65);marginal_error=float(np.max(abs(marginal-data['total'])));mass_error=abs(float(probs.sum())-(1-float(tail)));labels=sum((occup[:,i]>0).astype(int)<<i for i in range(m));click_finite=np.bincount(labels,weights=probs,minlength=1<<m);click_error=float(np.max(abs(click_finite-data['click'])));assert max(marginal_error,mass_error)<2e-12;assert click_error<=float(tailbound)+2e-12
        bm=mp.matrix([[mp.mpc(complex(x)) for x in row] for row in B]);lm=[mp.mpc(complex(x)) for x in linear];vacamp=mp.exp(mp.mpc(complex(lognorm)))
        @functools.lru_cache(None)
        def pairs(labels):
            if not labels:return mp.mpc(1)
            first=labels[0];rest=labels[1:];v=lm[first]*pairs(rest)
            for k,other in enumerate(rest):v+=bm[first,other]*pairs(rest[:k]+rest[k+1:])
            return v
        eligible=np.flatnonzero(totals<=10);chosen=np.unique(np.r_[0,rng.choice(eligible,size=128,replace=False)]);checks=[]
        for index in chosen:
            occupation=occup[index].astype(int);labels=tuple(i for i,k in enumerate(occupation) for _ in range(k));reference=abs(vacamp*pairs(labels))**2/mp.fprod(mp.factorial(int(k)) for k in occupation);error=abs(mp.mpf(float(probs[index]))-reference);assert error<mp.mpf('3e-15');checks.append(dict(key=int(keys[index]),reference=str(reference),observed=float(probs[index]),absolute_error=str(error)))
        row.update(full_pnr_patterns=len(keys),mass_error=mass_error,total_marginal_max_error=marginal_error,click_fock_max_error=click_error,high_precision_checks=len(checks));save(out,f'pair_checks_{j}.json',checks);pairs.cache_clear()
    if j==3:
        ref=np.load(ROOT/'results/iteration88_small_pnr_reference_v1/case_2.npz');meta=read(ROOT/'results/iteration88_small_pnr_reference_v1/case_2.json');difference=float(np.max(abs(data['click']-ref['target_click'])));assert difference<2e-14;data.update(pnr=ref['target']);row.update(reference88_sha256=H(ROOT/'results/iteration88_small_pnr_reference_v1/case_2.npz'),click_reference88_max_error=difference,pnr_tail=meta['tail_mass'],tail_chernoff=meta['tail_upper'])
    np.savez_compressed(out/f'law_{j}.npz',**data);rows.append(row);save(out,f'row_{j}.json',row);print(row,flush=True)
save(out,'validation.json',dict(passed=True,rows=rows,scope=__doc__));manifest(out)
