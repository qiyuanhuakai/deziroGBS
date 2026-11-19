"""Create a distinct exact-classical-completion sampler; keep v1 unchanged."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'native/gaussian_pnr_structured74_v1.cpp').read_text();s=s[s.index('API int pnrs74_sample('):]
s=s.replace('pnrs74_sample','pnrc74_sample').replace('std::fill(stats,stats+9,0)','std::fill(stats,stats+12,0)').replace('std::vector<Stats73> local(workers);','std::vector<Stats73> local(workers);std::vector<Classical74> tails(workers);')
s=s.replace('std::normal_distribution<double> normal;State73','std::mt19937_64 classical_gaussian(mix(seed+0x584b219c7d03e6a1ULL+row)),classical_poisson(mix(seed+0x1c6f94ae5832b07dULL+row));std::normal_distribution<double> normal;State73')
s=s.replace('if(state.factors.size()>std::size_t(retained)){','bool compressed=false;if(state.factors.size()>std::size_t(retained)){compressed=true;')
s=s.replace('++value;++photons;}','++value;++photons;if(compressed&&complete_classical74(c,state,classical_gaussian,classical_poisson,output+std::size_t(row)*m,photons,tails[id]))break;}')
s=s.replace('for(auto& e:failures)if(e)', 'for(auto& t:tails){stats[9]+=t.completed;stats[10]+=t.photons;stats[11]+=t.checks;}for(auto& e:failures)if(e)')
p=ROOT/'native/gaussian_pnr_classical74_v1.cpp';assert b'pnrc74_sample' not in p.read_bytes();p.write_bytes(p.read_bytes()+s.encode())
wrapper=(ROOT/'src/gbslab/native_structured_pnr.py').read_text().replace('NativeStructuredPNRSampler','NativeClassicalTailPNRSampler').replace('pnrs74_sample','pnrc74_sample').replace('stats=np.zeros(9)','stats=np.zeros(12)').replace('zip(STATS,stats.tolist())',"zip(STATS+('classical_completions','classical_photons','classicality_checks'),stats.tolist())")
wrapper=wrapper.replace('After the first compression,\n+the general Gaussian moment closure from iteration73 is used.'.replace('+',''),'After compression, positive-P Gaussian states finish by coherent-mixture\nPoisson sampling; other states use the general Gaussian moment closure.')
with (ROOT/'src/gbslab/native_classical_tail_pnr.py').open('xb') as f:f.write(wrapper.encode())
pilot=(ROOT/'experiments/iteration74_structured_pilot_v1.py').read_text().replace('native_structured_pnr','native_classical_tail_pnr').replace('NativeStructuredPNRSampler','NativeClassicalTailPNRSampler')
pilot=pilot.replace("('gaussian_pnr_structured74_v1.cpp'","('gaussian_pnr_classical74_v1.cpp','gaussian_pnr_structured74_v1.cpp'").replace("OUT/'structured.dll'","OUT/'classical.dll'")
with (ROOT/'experiments/iteration74_classical_pilot_v1.py').open('xb') as f:f.write(pilot.encode())
