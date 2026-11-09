"""All-history K=1 rounding bound and deterministic second-moment witness."""
import sys,json,math,hashlib
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')];sys.set_int_max_str_digits(0)
from iteration33_run import encode
from verify_iteration34 import decode
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=Path(__file__).parent;dest=out/'proof.json';assert not dest.exists();deps={}
    def read(n):
        p=ROOT/n;deps[n]=sha(p);return decode(json.loads(p.read_text()))
    raw=read('results/iteration39_global/case_00.stdout.json');build=read('results/iteration39_build/build.json');mom=read('results/iteration36_recovery/case_11.audit.json')['moments'];base=read('results/iteration35_panel/case_07.audit.json');source=ROOT/'third_party/FrugalGBS/include/fgbs/cpu/cpu_sampler.hpp';deps[source.relative_to(ROOT).as_posix()]=sha(source);assert raw['cutoff']==1 and raw['modes']==144 and raw['audit_identical_outputs'] and raw['threads']==1;assert build['precision']=='binary64' and build['body_hashes']['generate_sample']=='9c14a870f2875e808025d9e3535a71ba86144e5639ab14f0b7be5b594dabb3fb';assert '-ffast-math' not in build['command'];u=F(1,1<<53);epsilon=u+F(1,1<<64);operation_error=2*u+u*u;records=[]
    for mode,a in enumerate(raw['audit_paths'][0]):
        prev,z=map(float.fromhex,a[2:]);assert 2.**-1022<prev<=1 and 2.**-1022<z<1
        left=(F(z)+F(math.nextafter(z,-math.inf)))/2;right=(F(z)+F(math.nextafter(z,math.inf)))/2;lo=left/F(prev);hi=right/F(prev);assert F(1,8)<lo<=hi<F(7,8);p=(lo+hi)/2;radius=(hi-lo)/2;records.append(dict(mode=mode,previous=F(prev),rounded_product=F(z),fixed_p_interval=(lo,hi),proxy_zero=p,threshold_error_upper=radius+operation_error,finite_word_error_upper=radius+operation_error+epsilon))
    # The induction is uniform over every binary history, not only this path.
    zero_factor=F(1,8)*(1-u);one_factor=(1-F(7,8)*(1+u))*(1-u);upper_one=(1-F(1,8)*(1-u))*(1+u);assert zero_factor>F(1,16) and one_factor>F(1,16) and upper_one<1 and F(7,8)*(1+u)<1;assert F(1,1<<579)>F(1,1<<1022)
    coupling=sum(r['finite_word_error_upper'] for r in records);ps=[r['proxy_zero'] for r in records];mean=sum((1-p for p in ps),F(0));variance=sum((p*(1-p) for p in ps),F(0));center=72;scale=5184;proxy_y=(variance+(mean-center)**2)/scale;mu=mom['mean_interval'];second=mom['second_moment_interval'];target=((second[0]-144*mu[1]+5184)/5184,(second[1]-144*mu[0]+5184)/5184);gap=max(F(0),target[0]-proxy_y,proxy_y-target[1]);lower=max(F(0),gap-coupling);transfer=base['transfers']['-40']['all_measurement_TV_transfer_upper'];result=dict(source_sha256=sha(Path(__file__)),dependencies=deps,records=records,unit_roundoff=u,uniform_CDF_error_upper=epsilon,multiply_divide_error_upper=operation_error,all_history_previous_lower=F(1,1<<576),minimum_intermediate_lower=F(1,1<<579),zero_update_factor_lower=zero_factor,one_update_factor_lower=one_factor,one_update_factor_upper=upper_one,proxy_mean=mean,proxy_variance=variance,center=center,scale=scale,proxy_observable_mean=proxy_y,target_observable_interval=target,full_law_coupling_upper=coupling,full_TV_deterministic_lower=lower,shifted_target_TV_deterministic_lower=max(F(0),lower-transfer),assumptions=['Independent uniform uint64 words with the inspected GCC mapping','Recorded binary64 normal multiplication and division under nearest rounding','Pinned K=1 source expression; fixed per-mode p and no fast math','Physical target intervals and transfer reused from rounds35/36'],scope='Post-cohort deterministic analytic bound; no confidence budget or PRNG independence claim')
    dest.write_text(json.dumps(encode(result),indent=2)+'\n');print(json.dumps({k:float(result[k]) for k in ('proxy_mean','proxy_variance','full_law_coupling_upper','full_TV_deterministic_lower','shifted_target_TV_deterministic_lower')}))
if __name__=='__main__':main()
