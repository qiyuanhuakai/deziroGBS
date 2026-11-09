import math
import unittest
import mpmath as mp
import numpy as np
from scipy.stats import nbinom
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from gbslab.mps_probe import schmidt_parameters,mps_to_dense
from gbslab.gaussian_mps_feasibility import marginal_photon_tails,equal_core_spectra,representation_bounds,compress_box_once


class FeasibilityTests(unittest.TestCase):
    def test_thermal_tail(self):
        n=.3; dims=(4,8,12)
        result=marginal_photon_tails(n,0,dims)
        np.testing.assert_allclose(result['tail_upper_estimates'],[(n/(1+n))**d for d in dims],rtol=1e-12,atol=1e-30)

    def test_squeezed_vacuum_tail(self):
        s=.4; dims=(5,6,8,12); n=math.sinh(s)**2
        result=marginal_photon_tails(n,math.sinh(s)*math.cosh(s),dims)
        expected=[nbinom.sf((d-1)//2,.5,1/math.cosh(s)**2) for d in dims]
        np.testing.assert_allclose(result['tail_upper_estimates'],expected,rtol=1e-12,atol=1e-30)

    def test_independent_generating_derivatives(self):
        n=.4; s=.25; result=marginal_photon_tails(n,s,(4,8))
        with mp.workdps(60):
            def g(z): return ((1+mp.mpf(n)*(1-z))**2-mp.mpf(s)**2*(1-z)**2)**(-mp.mpf('.5'))
            for k in range(15): self.assertAlmostEqual(result['probabilities'][k],float(mp.diff(g,0,k)/math.factorial(k)),delta=2e-15)

    def test_cross_spectra_against_symplectic(self):
        for m in (2,4,8):
            u=haar_unitary(m,np.random.default_rng(1900+m)); result=equal_core_spectra(u,.4,32)
            V=covariance_from_interferometer(u,.4,1.)
            for c in result['cuts']: np.testing.assert_allclose(np.sort(c['parameters']),np.sort(schmidt_parameters(V,range(c['cut']))),atol=2e-14,rtol=0)

    def test_box_tt_bound_for_arbitrary_states(self):
        rng=np.random.default_rng(1904)
        for _ in range(4):
            psi=rng.normal(size=(4,4,4,4))+1j*rng.normal(size=(4,4,4,4)); psi/=np.linalg.norm(psi)
            box=psi[:3,:3,:3,:3]; box_tail=1-float(np.linalg.norm(box)**2)
            for rank in (1,2,4):
                tails=[float(np.sum(np.linalg.svd(psi.reshape(4**k,-1),compute_uv=False)[rank:]**2)) for k in range(1,4)]
                tensors,info=compress_box_once(box,rank); state=mps_to_dense(tensors)
                actual2=1-abs(np.vdot(box,state))**2
                self.assertLessEqual(actual2,box_tail+sum(tails)+1e-13)
                self.assertAlmostEqual(actual2,info['original_state_trace_from_orthogonal_loss']**2,delta=2e-14)

    def test_product_core_no_entanglement(self):
        result=representation_bounds(np.eye(4),.3,dimensions=(6,8),ranks=(1,2))
        for row in result['rows']: self.assertEqual(row['schmidt_tail_sum'],0)

    def test_conditional_losses_are_not_ideal_losses(self):
        # Rotating rank-one projections: each conditional loss tiny, final orthogonal.
        N=100; state=np.array([1.,0.]); initial=state.copy(); conditional=0.; ideal=0.
        for k in range(1,N+1):
            direction=np.array([math.cos(k*math.pi/(2*N)),math.sin(k*math.pi/(2*N))])
            projector=np.outer(direction,direction); output=projector@state
            conditional+=1-float(output@output); ideal+=1-float(initial@projector@initial)
            state=output/np.linalg.norm(output)
        self.assertLess(math.sqrt(conditional),.16)
        self.assertAlmostEqual(math.sqrt(1-float(initial@state)**2),1)
        self.assertGreater(ideal,1)

    def test_vacuum_and_invalid_inputs(self):
        result=marginal_photon_tails(0,0,(4,8)); np.testing.assert_array_equal(result['tail_upper_estimates'],[0,0])
        with self.assertRaises(ValueError): marginal_photon_tails(.1,1,(4,))
        with self.assertRaises(ValueError): marginal_photon_tails(.1,0,(128,))
        with self.assertRaises(ValueError): equal_core_spectra(np.ones((3,3)),.4,32)

if __name__=='__main__': unittest.main()
