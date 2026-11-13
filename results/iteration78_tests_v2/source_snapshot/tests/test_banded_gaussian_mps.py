from functools import lru_cache
import math
import unittest
import numpy as np
from gbslab.banded_gaussian_mps import build_banded_box,build_banded_core,truncate_band,pairing_covariance,pairing_distance,pairing_log_normalization,frontier_element_bound
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from gbslab.mps_probe import mps_to_dense


def reference_box(B,d):
    m=len(B); normalization=math.exp(pairing_log_normalization(B))
    @lru_cache(None)
    def amplitude(n):
        if not any(n): return complex(normalization)
        if sum(n)%2: return 0j
        i=next(j for j,x in enumerate(n) if x); value=0j
        for j in range(m):
            count=n[j]-int(i==j)
            if count>0:
                sub=list(n); sub[i]-=1; sub[j]-=1
                value+=B[i,j]*math.sqrt(count)*amplitude(tuple(sub))
        return value/math.sqrt(n[i])
    out=np.empty((d,)*m,complex)
    for n in np.ndindex(out.shape): out[n]=amplitude(n)
    return out


class BandedMpsTests(unittest.TestCase):
    def test_frontier_against_creation_recurrence(self):
        u=haar_unitary(4,np.random.default_rng(2001)); original=-math.tanh(.4)*(u@u.T)
        for width in (0,1,2,3):
            B=truncate_band(original,width); tensors,info=build_banded_box(B,5,width)
            expected=reference_box(B,5); actual=mps_to_dense(tensors)*math.sqrt(info['box_mass'])
            np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=0)
            self.assertAlmostEqual(info['box_mass'],float(np.linalg.norm(expected)**2),delta=2e-13)

    def test_pair_number_correlations(self):
        t=.3; B=np.array([[0,t],[t,0]],complex); tensors,info=build_banded_box(B,6,1)
        expected=np.diag([math.sqrt(1-t*t)*t**k for k in range(6)])
        np.testing.assert_allclose(mps_to_dense(tensors)*math.sqrt(info['box_mass']),expected,atol=2e-15)

    def test_covariance_against_uniform_optics(self):
        u=haar_unitary(5,np.random.default_rng(2002)); B=-math.tanh(.4)*(u@u.T)
        np.testing.assert_allclose(pairing_covariance(B),covariance_from_interferometer(u,.4,1.),atol=2e-14,rtol=0)

    def test_fidelity_independent_covariance(self):
        u=haar_unitary(4,np.random.default_rng(2003)); B=-math.tanh(.4)*(u@u.T)
        for w in (0,1,2,3):
            C=truncate_band(B,w); V=pairing_covariance(B); W=pairing_covariance(C)
            fidelity=np.linalg.det((V+W)/2)**(-.5)
            self.assertAlmostEqual(pairing_distance(B,C)**2,max(0.,1-fidelity),delta=2e-14)

    def test_compressed_budget_against_original_gaussian(self):
        u=haar_unitary(4,np.random.default_rng(2004)); B=-math.tanh(.4)*(u@u.T); original=reference_box(B,5)
        for w in (1,2,3):
            tensors,info=build_banded_core(B,5,w,3); actual=mps_to_dense(tensors)
            distance=math.sqrt(max(0.,1-abs(np.vdot(original,actual))**2))
            self.assertLessEqual(distance,info['construction_trace_budget']+1e-12)
            band=reference_box(truncate_band(B,w),5)
            self.assertAlmostEqual(1-abs(np.vdot(band,actual))**2,info['distance_to_band_gaussian']**2,delta=2e-13)

    def test_diagonal_vacuum_and_frontier_caps(self):
        tensors,info=build_banded_box(np.zeros((8,8)),4,0)
        self.assertEqual(info['box_mass'],1); self.assertEqual(info['frontier_elements'],32)
        self.assertEqual(frontier_element_bound(8,4,0),32)
        with self.assertRaises(MemoryError): build_banded_box(np.zeros((32,32)),6,4,max_elements=1000)
        with self.assertRaises(ValueError): build_banded_box(np.eye(4)*1.1,4,0)
        with self.assertRaises(ValueError): build_banded_box(np.ones((4,4))*.05,4,1)

    def test_output_phase_covariance(self):
        u=haar_unitary(4,np.random.default_rng(2005)); B=truncate_band(-.3*(u@u.T),2); phase=np.exp(1j*np.arange(4))
        first,info=build_banded_box(B,4,2); second,other=build_banded_box(B*phase[:,None]*phase[None,:],4,2)
        a=mps_to_dense(first); b=mps_to_dense(second); factors=np.ones((4,)*4,complex)
        for i in range(4):
            shape=[1]*4; shape[i]=4; factors*=phase[i]**np.arange(4).reshape(shape)
        np.testing.assert_allclose(b,a*factors,atol=1e-13,rtol=0)

if __name__=='__main__': unittest.main()
