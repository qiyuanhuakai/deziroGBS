import ctypes as C
import os
from pathlib import Path
import shutil
import unittest
import numpy as np
from gbslab.native_gaussian_reset import NativeGaussianResetSampler,lossy_squeezed_components,DP,UP


class GaussianResetDecompositionTests(unittest.TestCase):
    def test_exact_input_covariance_identity(self):
        for r in (0.,.3,.7,1.1,3.):
            for eta in (0.,.01,.3,.65,1.):
                core,noise=lossy_squeezed_components(2,r,eta)
                target=np.diag(np.r_[np.full(2,.5*eta*np.expm1(2*r)),np.full(2,.5*eta*np.expm1(-2*r))])
                np.testing.assert_allclose(core+np.diag(noise),target,rtol=2e-13,atol=2e-14)
                self.assertTrue(np.all(noise>=0))

    def test_reject_invalid_decomposition_parameters(self):
        for args in [(0,.7,.5),(True,.7,.5),(2,-.1,.5),(2,np.nan,.5),(2,.7,1.1),(2,.7,np.inf)]:
            with self.assertRaises(ValueError):lossy_squeezed_components(*args)


@unittest.skipUnless(os.environ.get('GBSLAB_GAUSSIAN_RESET_LIBRARY'),'explicit native Gaussian reset library required')
class NativeGaussianResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=Path(os.environ['GBSLAB_GAUSSIAN_RESET_LIBRARY'])
        compiler=shutil.which('g++');cls.runtime=[Path(compiler).parent] if compiler else []
        cls.blas=os.environ.get('GBSLAB_GAUSSIAN_RESET_BLAS')

    def model(self,**kwargs):
        v=kwargs.pop('v',np.eye(2,dtype=np.complex128));x=kwargs.pop('x',np.zeros((4,4)))
        return NativeGaussianResetSampler(v,x,self.library,runtime_dirs=self.runtime,**kwargs)

    def test_vacuum_all_selectors(self):
        with self.model() as model:
            for selector in ('scalar','eigen','leverage','diagonal','anchor'):
                result=model.sample(256,69,selector=selector)
                self.assertFalse(np.any(result['counts']))
                self.assertEqual(result['statistics']['vacuum_samples'],256)

    def test_coherent_poisson_moments(self):
        mean=np.array([1.,.7,-.2,.4]);expected=(mean[:2]**2+mean[2:]**2)/2
        with self.model(mean=mean) as model:
            counts=model.sample(40000,691901,selector='anchor')['counts']
        self.assertTrue(np.all(abs(counts.mean(axis=0)-expected)<=7*np.sqrt(expected/len(counts))))
        self.assertLess(abs(np.cov(counts.T)[0,1]),.025)

    def test_seed_reproduction_and_noise_copy(self):
        x,variance=lossy_squeezed_components(2,.7,.65);copy=variance.copy()
        with self.model(x=x,noise_variance=variance) as model:
            variance[:]=0
            np.testing.assert_array_equal(model.noise_variance,copy)
            for selector in ('eigen','diagonal','anchor'):
                a=model.sample(128,691902,selector=selector)['counts'];b=model.sample(128,691902,selector=selector)['counts']
                np.testing.assert_array_equal(a,b)
                self.assertGreater(a.sum(),0)

    def test_empty_batch_and_lifetime(self):
        model=self.model();self.assertEqual(model.sample(0,0)['counts'].shape,(0,2));model.close();model.close()
        with self.assertRaises(RuntimeError):model.sample(1,0)

    def test_invalid_models(self):
        cases=[dict(v=np.ones((2,2))),dict(v=np.ones((1,2))),dict(x=-np.eye(4)),dict(x=np.eye(3)),dict(x=np.full((4,4),np.nan)),dict(x=np.eye(4,dtype=complex)),dict(mean=[1,2]),dict(mean=[1,2,3,np.inf]),dict(noise_variance=[0,0,0,-1])]
        bad=np.zeros((4,4));bad[0,1]=.01;cases.append(dict(x=bad))
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):self.model(**kwargs)

    def test_invalid_sample_arguments(self):
        with self.model() as model:
            for count,seed in [(-1,0),(True,0),(2**31,0),(1,-1),(1,2**64),(1,True)]:
                with self.assertRaises(ValueError):model.sample(count,seed)
            with self.assertRaises(ValueError):model.sample(1,0,selector='invalid')
            with self.assertRaises(ValueError):model.sample(1,0,selector='openblas')

    def test_precision_restored_on_success_and_native_error(self):
        with self.model() as model:
            model.lib.gr_precision.argtypes=[DP];model.lib.gr_precision.restype=None
            before=np.zeros(5);after=np.zeros(5);model.lib.gr_precision(before.ctypes.data_as(DP));model.sample(2,0)
            model.mean[0]=np.nan
            with self.assertRaises(ArithmeticError):model.sample(1,0)
            model.lib.gr_precision(after.ctypes.data_as(DP))
            self.assertEqual(before[0],after[0]);self.assertEqual(before[0],before[4]);self.assertEqual(after[0],after[4]);self.assertGreater(after[3],0)

    def test_openblas_rates_and_thread_restore(self):
        if not self.blas:self.skipTest('explicit ILP64 OpenBLAS required')
        x,variance=lossy_squeezed_components(2,.7,.65)
        rng=np.random.default_rng(691903);v=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)))[0])
        with self.model(v=v,x=x,noise_variance=variance,blas_library=self.blas,threads=1) as model:
            old=model.getter()
            try:
                model.setter(2);a=model.sample(1024,691904,selector='eigen')['counts'];self.assertEqual(model.getter(),2)
                b=model.sample(1024,691904,selector='openblas')['counts'];np.testing.assert_array_equal(a,b);self.assertEqual(model.getter(),2)
                model.mean[0]=np.nan
                with self.assertRaises(ArithmeticError):model.sample(1,0,selector='openblas')
                self.assertEqual(model.getter(),2)
            finally:model.setter(old)

if __name__=='__main__':unittest.main()
