import ctypes as C
import os
from pathlib import Path
import shutil
import unittest
import numpy as np
from gbslab.native_gaussian_reset import lossy_squeezed_components, DP
from gbslab.native_gaussian_reset_parallel import NativeParallelGaussianResetSampler


@unittest.skipUnless(os.environ.get('GBSLAB_GAUSSIAN_RESET_PARALLEL_LIBRARY'),
                     'explicit parallel Gaussian-reset library required')
class NativeParallelGaussianResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=Path(os.environ['GBSLAB_GAUSSIAN_RESET_PARALLEL_LIBRARY'])
        cls.runtime=[Path(shutil.which('g++')).parent]
        cls.blas=os.environ.get('GBSLAB_GAUSSIAN_RESET_BLAS')

    def model(self,**kwargs):
        v=kwargs.pop('v',np.eye(2,dtype=np.complex128))
        x=kwargs.pop('x',np.zeros((4,4)))
        return NativeParallelGaussianResetSampler(v,x,self.library,
                                                  runtime_dirs=self.runtime,**kwargs)

    def test_vacuum_across_workers_and_selectors(self):
        with self.model() as model:
            for selector in ('scalar','eigen','leverage','diagonal','anchor'):
                for workers in (1,4,16):
                    result=model.sample(32,701,selector=selector,workers=workers)
                    self.assertFalse(result['counts'].any())
                    self.assertEqual(result['statistics']['completed_samples'],32)
                    self.assertEqual(result['statistics']['vacuum_samples'],32)
                    self.assertEqual(result['statistics']['workers'],workers)

    def test_displaced_counts_independent_of_worker_schedule(self):
        rng=np.random.default_rng(707001)
        v=np.linalg.qr(rng.normal(size=(8,3))+1j*rng.normal(size=(8,3)))[0]
        x,variance=lossy_squeezed_components(3,.9,.65)
        with self.model(v=v,x=x,noise_variance=variance) as model:
            for selector in ('eigen','diagonal','anchor'):
                reference=model.sample(1001,707002,selector=selector,workers=1)
                for workers in (4,16):
                    result=model.sample(1001,707002,selector=selector,workers=workers)
                    np.testing.assert_array_equal(result['counts'],reference['counts'])
                    for key in ('photons','proposals','vacuum_samples','maximum_photons',
                                'survival_evaluations','newton_fallbacks','flow_calls'):
                        self.assertEqual(result['statistics'][key],reference['statistics'][key])

    def test_newton_matches_bisection_fixture(self):
        x,variance=lossy_squeezed_components(2,1.1,.65)
        with self.model(x=x,noise_variance=variance,workers=4) as model:
            old=model.sample(2048,707003,flow='bisection')
            new=model.sample(2048,707003,flow='newton')
            np.testing.assert_array_equal(old['counts'],new['counts'])
            self.assertLess(new['statistics']['survival_evaluations'],old['statistics']['survival_evaluations'])

    def test_coherent_independent_poisson_moments(self):
        mean=np.array([1.,.7,-.2,.4]);expected=(mean[:2]**2+mean[2:]**2)/2
        with self.model(mean=mean,workers=8) as model:
            counts=model.sample(40000,707004,selector='anchor')['counts']
        self.assertTrue(np.all(abs(counts.mean(axis=0)-expected)<=7*np.sqrt(expected/len(counts))))
        self.assertLess(abs(np.cov(counts.T)[0,1]),.025)

    def test_zero_batch_and_more_workers_than_rows(self):
        with self.model(workers=256) as model:
            empty=model.sample(0,0)
            self.assertEqual(empty['counts'].shape,(0,2))
            self.assertEqual(empty['statistics']['completed_samples'],0)
            self.assertEqual(empty['statistics']['workers'],0)
            self.assertEqual(model.sample(2,0)['statistics']['workers'],2)

    def test_argument_validation(self):
        for workers in (0,257,-1,True,1.5):
            with self.assertRaises(ValueError):self.model(workers=workers)
        with self.assertRaises(ValueError):self.model(threads=2)
        with self.model() as model:
            for kwargs in ({'workers':0},{'workers':True},{'flow':'bad'},
                           {'backend':'bad'},{'backend':'lapack'},
                           {'selector':'openblas'},{'selector':'bad'},
                           {'flow':'bisection','backend':'lapack'}):
                with self.assertRaises(ValueError):model.sample(1,0,**kwargs)
            for count,seed in ((-1,0),(True,0),(2**31,0),(1,-1),(1,2**64),(1,True)):
                with self.assertRaises(ValueError):model.sample(count,seed)
        with self.assertRaises(RuntimeError):model.sample(1,0)

    def test_caller_precision_after_native_failure(self):
        with self.model(workers=4) as model:
            model.lib.gr_precision.argtypes=[DP]
            before=np.zeros(5);after=np.zeros(5)
            model.lib.gr_precision(before.ctypes.data_as(DP))
            model.mean[0]=np.nan
            with self.assertRaises(ArithmeticError):model.sample(10,0)
            model.lib.gr_precision(after.ctypes.data_as(DP))
            self.assertEqual(before[0],after[0])
            model.mean[0]=0
            self.assertFalse(model.sample(10,0)['counts'].any())

    def test_lapack_replay_and_blas_restore(self):
        if not self.blas:self.skipTest('explicit ILP64 OpenBLAS required')
        x,variance=lossy_squeezed_components(2,.7,.65)
        with self.model(x=x,noise_variance=variance,blas_library=self.blas,workers=4) as model:
            old=model.getter()
            try:
                model.setter(2)
                a=model.sample(4096,707005,backend='eigen')
                self.assertEqual(model.getter(),2)
                b=model.sample(4096,707005,backend='lapack')
                np.testing.assert_array_equal(a['counts'],b['counts'])
                self.assertEqual(model.getter(),2)
            finally:model.setter(old)

    def test_worker_exception_join_and_recovery(self):
        if not self.blas:self.skipTest('explicit ILP64 OpenBLAS required')
        with self.model(mean=[2.,1.,0.,0.],blas_library=self.blas,workers=4) as model:
            old=model.getter()
            try:
                model.setter(2)
                model.lib.gr_set_gemm(model.handle,None)
                with self.assertRaisesRegex(ArithmeticError,'GEMM'):
                    model.sample(100,707006,selector='openblas')
                self.assertEqual(model.getter(),2)
                model.lib.gr_set_gemm(model.handle,C.cast(model.blas.scipy_cblas_zgemm64_,C.c_void_p))
                self.assertGreater(model.sample(100,707006,selector='openblas')['counts'].sum(),0)
                self.assertEqual(model.getter(),2)
            finally:model.setter(old)


if __name__=='__main__':unittest.main()
