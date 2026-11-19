from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import mpmath as mp
from gbslab.sampling_audit import choice_intervals, enumerate_floating_law, high_precision_law
from gbslab.local_mps import local_circuit, build_local_core_layerwise, sample_local_threshold
from gbslab.mps_probe import fixed_threshold_law


class SamplingAuditTests(unittest.TestCase):
    def test_tree_counts_against_exhaustive_sampler_grid(self):
        gates,_=local_circuit(2,2,np.random.default_rng(1306))
        tensors,_=build_local_core_layerwise(2,.4,4,4,gates)
        alpha=np.array([[.2+.3j,-.4j]])
        bits=3; grid=2**bits
        class GridRNG:
            def __init__(self,uniforms): self.uniforms=iter(uniforms)
            def choice(self,size,p):
                cdf=p.cumsum(); cdf/=cdf[-1]
                return cdf.searchsorted(next(self.uniforms),side='right')
        counts=[0]*4
        for i in range(grid):
            for j in range(grid):
                samples,_=sample_local_threshold(tensors,alpha,GridRNG([i/grid,j/grid]))
                counts[int(samples[0,0])+2*int(samples[0,1])]+=1
        r=enumerate_floating_law(tensors,alpha[0],bits=bits)
        self.assertEqual(counts,r['grid_numerators'])

    def test_grid_counts_against_exhaustive_search(self):
        for p in ([0,.25,0,.75],[1e-30,.3,.7],[.1,.2,.3,.4]):
            for bits in (1,3,8):
                widths,counts,cdf=choice_intervals(p,bits)
                observed=np.bincount(cdf.searchsorted(np.arange(2**bits)/2**bits,side='right'),minlength=len(p))
                self.assertEqual(counts,observed.tolist())
                self.assertEqual(sum(widths),1)

    def test_choice_matches_numpy_same_uniforms(self):
        p=np.array([1e-12,.1,.2,.699999999999])
        _,_,cdf=choice_intervals(p)
        a=np.random.default_rng(1301); b=np.random.default_rng(1301)
        np.testing.assert_array_equal(a.choice(len(p),size=2000,p=p),cdf.searchsorted(b.random(2000),side='right'))

    def test_vacuum_exact(self):
        tensors=[np.array([1,0,0,0],complex).reshape(1,4,1)]*2
        r=enumerate_floating_law(tensors,[0,0])
        self.assertEqual(r['real_cdf_law'],[1,0,0,0])
        self.assertEqual(r['grid_numerators'],[r['grid_denominator'],0,0,0])

    def test_rare_event_grid_floor(self):
        tensors=[np.array([1,0],complex).reshape(1,2,1)]
        r=enumerate_floating_law(tensors,[8])
        self.assertEqual(r['grid_numerators'][0],1)
        with mp.workdps(100):
            ref=high_precision_law(tensors,[8],digits=90)
            self.assertLess(abs(mp.mpf(ref['law_decimal'][0])-mp.exp(-64)),mp.mpf('1e-95'))
        self.assertLess(float(r['quantization_tv']),2**-53)

    def test_failure_keeps_full_mass(self):
        r=enumerate_floating_law([np.array([2,0],complex).reshape(1,2,1)],[0])
        self.assertEqual(r['real_failure'],1)
        self.assertEqual(r['grid_failure_numerator'],r['grid_denominator'])

    def test_node_cap_aborts(self):
        with self.assertRaises(MemoryError):
            enumerate_floating_law([np.array([1,0],complex).reshape(1,2,1)],[1],max_nodes=1)

    def test_invalid_input(self):
        for bits in (0,54,True):
            with self.assertRaises(ValueError): choice_intervals([1],bits)
        for p in ([0,0],[-1,2],[float('nan')]):
            with self.assertRaises(ValueError): choice_intervals(p)

    def test_entangled_reference_and_convergence(self):
        gates,_=local_circuit(3,3,np.random.default_rng(1302))
        tensors,_=build_local_core_layerwise(3,.3,4,8,gates)
        alpha=np.array([.2+.1j,-.3j,.4])
        low=high_precision_law(tensors,alpha,digits=60)
        high=high_precision_law(tensors,alpha,digits=90)
        np.testing.assert_allclose(np.array(high['law_decimal'],float),fixed_threshold_law(tensors,alpha),atol=1e-14)
        with mp.workdps(100):
            self.assertLess(max(abs(mp.mpf(x)-mp.mpf(y)) for x,y in zip(low['law_decimal'],high['law_decimal'])),mp.mpf('1e-55'))
        r=enumerate_floating_law(tensors,alpha)
        self.assertEqual(r['real_failure'],0)
        self.assertLess(sum(abs(float(p)-float(q)) for p,q in zip(r['real_cdf_law'],high['law_decimal'])),1e-13)

    def test_refactor_preserves_archived_sampler(self):
        root=Path(__file__).resolve().parents[1]
        path=root/'results/iteration12_layerwise_quality/source_snapshot/src/gbslab/local_mps.py'
        spec=importlib.util.spec_from_file_location('gbslab.local_mps_iteration12',path)
        old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        gates,_=local_circuit(4,4,np.random.default_rng(1303))
        tensors,_=build_local_core_layerwise(4,.3,4,4,gates)
        alpha=np.random.default_rng(1304).normal(size=(100,4)).astype(complex)*(.3+.2j)
        a,ia=sample_local_threshold(tensors,alpha,np.random.default_rng(1305))
        b,ib=old.sample_local_threshold(tensors,alpha,np.random.default_rng(1305))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(ia,ib)


if __name__=='__main__': unittest.main()
