import ctypes as C
import math
import os
from pathlib import Path
import shutil
import unittest
import numpy as np
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler,DP

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('GBSLAB_POLYNOMIAL_PNR_LIBRARY'),'explicit polynomial PNR library required')
class PolynomialPNRTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativePolynomialPNRSampler(kwargs.pop('v',np.eye(2,dtype=complex)),
            kwargs.pop('x',np.zeros((4,4))),os.environ['GBSLAB_POLYNOMIAL_PNR_LIBRARY'],
            runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)

    def test_vacuum_empty_and_closed(self):
        with self.model(retained=3) as model:
            r=model.sample(128,736101)
            self.assertFalse(r['counts'].any());self.assertEqual(r['counts'].dtype,np.uint32)
            self.assertEqual(r['statistics']['completed_samples'],128)
            self.assertEqual(r['statistics']['root_iterations'],0)
            self.assertEqual(model.sample(0,0)['counts'].shape,(0,2))
        with self.assertRaises(RuntimeError):model.sample(1,0)

    def test_thermal_photon_subtracted_moments(self):
        # k subtractions from a thermal mode: mass k!*n^k, mean n*(k+1).
        x=np.eye(2)*.6;mean=np.zeros(2)
        with self.model(v=np.ones((1,1),complex),x=x) as model:
            probe=model.lib.pnr74_probe;probe.argtypes=[C.c_void_p,DP,DP,C.c_int,C.POINTER(C.c_int),DP,DP,DP,C.c_char_p,C.c_int]
            for k in (1,2,3,4):
                modes=np.zeros(k,np.int32);mass=np.zeros(1);out=np.zeros((2,2));w=np.zeros(2);error=C.create_string_buffer(4096)
                rc=probe(model.handle,x.ctypes.data_as(DP),mean.ctypes.data_as(DP),k,modes.ctypes.data_as(C.POINTER(C.c_int)),mass.ctypes.data_as(DP),out.ctypes.data_as(DP),w.ctypes.data_as(DP),error,len(error))
                self.assertEqual(rc,0,error.value);self.assertAlmostEqual(mass[0],math.factorial(k)*.6**k,places=12)
                np.testing.assert_allclose(out,np.eye(2)*.6*(k+1),atol=1e-12,rtol=0);self.assertFalse(w.any())

    def test_independent_holdout_and_worker_replay(self):
        data=np.load(ROOT/'results/iteration73_polynomial_pilot_v2/input_3.npz');expected=np.load(ROOT/'results/iteration73_pnr_holdout_v1/counts_3_poly3_0.npz')['counts'][:1024]
        seed=0x75000000000000+40*2**40
        with self.model(v=data['v'],x=data['x'],noise_variance=data['variance'],retained=3) as model:
            a=model.sample(1024,seed,workers=1);b=model.sample(1024,seed,workers=16)
        np.testing.assert_array_equal(a['counts'],expected);np.testing.assert_array_equal(a['counts'],b['counts'])
        self.assertEqual(b['statistics']['photons'],b['counts'].sum())

    def test_documented_parity_boundary_fails_explicitly_and_recovers(self):
        # Known v1 limitation: a mathematically zero odd-sector vacuum mass
        # rounds slightly negative. This asserts disclosure, not support of it.
        with self.model(v=np.ones((1,1),complex),x=np.diag([-.25,.5]),retained=3) as model:
            with self.assertRaises(ArithmeticError):model.sample(1000,736001)
            model.noise_variance[:]=.2
            self.assertEqual(model.sample(128,736001)['counts'].shape,(128,1))

    def test_validation(self):
        for retained in (-1,4,True,1.5):
            with self.assertRaises(ValueError):self.model(retained=retained)
        with self.model() as model:
            for count,seed in ((True,0),(-1,0),(1,-1),(1,2**64)):
                with self.assertRaises(ValueError):model.sample(count,seed)
            model.mean[0]=np.nan
            with self.assertRaises(ArithmeticError):model.sample(1,0)

if __name__=='__main__':unittest.main()
