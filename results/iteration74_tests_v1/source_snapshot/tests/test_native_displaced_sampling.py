"""Native integration checks; explicit library environment enables each backend."""
import os
from pathlib import Path
import unittest
import numpy as np
from gbslab.displaced_sampling import DisplacedThresholdSampler
from gbslab.native_displaced_sampling import NativeDisplacedThresholdSampler

class NativeCases:
    @classmethod
    def setUpClass(cls):
        key='GBSLAB_NATIVE_'+cls.backend.upper()+'_LIBRARY';path=os.environ.get(key)
        if not path:raise unittest.SkipTest(key+' is not configured; build and validate native libraries explicitly')
        cls.library=Path(path);cls.runtime_dirs=[p for p in os.environ.get('GBSLAB_NATIVE_RUNTIME_DIRS','').split(os.pathsep) if p]
        cls.blas=os.environ.get('GBSLAB_NATIVE_BLAS_LIBRARY') if cls.backend=='cpu' else None
        if not cls.library.is_file():raise RuntimeError('Configured native library is missing')

    def sampler(self,tensors,**kwargs):
        return NativeDisplacedThresholdSampler(tensors,self.library,backend=self.backend,runtime_dirs=self.runtime_dirs,blas_library=self.blas,**kwargs)

    def test_entangled_trajectories_match_reference(self):
        tensors=[np.eye(2,dtype=complex).reshape(1,2,2)/np.sqrt(2),np.eye(2,dtype=complex).reshape(2,2,1)]
        rng=np.random.default_rng(680301);beta=.4*(rng.normal(size=(127,2))+1j*rng.normal(size=(127,2)));u=rng.random((127,2));expected=DisplacedThresholdSampler(tensors).sample(beta,u)
        with self.sampler(tensors,threads=2) as sampler:actual=sampler.sample(beta,u)
        np.testing.assert_array_equal(actual['refined_labels'],expected['refined_labels'])
        np.testing.assert_allclose(actual['selected_refined_probabilities'],expected['selected_refined_probabilities'],rtol=1e-12,atol=1e-14)

    def test_empty_batch_and_closed_model(self):
        with self.sampler([np.array([1.,0.]).reshape(1,2,1)]) as sampler:
            result=sampler.sample(np.empty((0,1),complex),np.empty((0,1)))
            self.assertEqual(result['clicks'].shape,(0,1));self.assertEqual(result['maximum_normalization_residual'],0.)
        with self.assertRaises(RuntimeError):sampler.sample([[0.]],[[.5]])
        sampler.close()

    def test_zero_displacement_structural_zeros(self):
        with self.sampler([np.array([1.,0.,0.]).reshape(1,3,1)]) as sampler:
            result=sampler.sample(np.zeros((3,1)),np.array([[0.],[.5],[np.nextafter(1.,0.)]]))
        np.testing.assert_array_equal(result['refined_labels'],0);np.testing.assert_array_equal(result['selected_refined_probabilities'],1.)

    def test_positive_tiny_branch_is_retained(self):
        with self.sampler([np.ones((1,1,1))]) as sampler:result=sampler.sample([[1e-8]],[[0.]])
        self.assertEqual(result['refined_labels'][0,0],1)
        self.assertGreater(result['selected_refined_probabilities'][0,0],0.)
        self.assertAlmostEqual(result['selected_refined_probabilities'][0,0]/1e-16,1.,places=12)

    def test_numerical_rejection_then_valid_reuse(self):
        with self.sampler([np.array([1.,0.]).reshape(1,2,1)]) as sampler:
            with self.assertRaises(ArithmeticError):sampler.sample([[1e-200]],[[.5]])
            self.assertEqual(sampler.sample([[0.]],[[.5]])['refined_labels'][0,0],0)

    def test_invalid_coordinates(self):
        with self.sampler([np.ones((1,1,1))]) as sampler:
            for beta,u in [([[float('nan')]],[[.5]]),([[0.]],[[1.]]),([[0.]],[[float('inf')]]),([[0.]],[[.5+1j]]),([[0.,0.]],[[.5,.5]])]:
                with self.assertRaises(ValueError):sampler.sample(beta,u)

    def test_default_normalization_and_explicit_norm_rejection(self):
        tensors=[np.array([2.,0.]).reshape(1,2,1)]
        with self.sampler(tensors) as sampler:self.assertEqual(sampler.sample([[0.]],[[.5]])['refined_labels'][0,0],0)
        with self.sampler(tensors,canonical=True) as sampler:
            with self.assertRaises(ArithmeticError):sampler.sample([[.2]],[[.5]])

    def test_invalid_model(self):
        for tensors in ([],[np.zeros((1,2,1))],[np.ones((2,2,1))],[np.array([float('nan')]).reshape(1,1,1)]):
            with self.assertRaises(ValueError):self.sampler(tensors)

    def test_local_dimension_limit_is_explicit(self):
        tensors=[np.ones((1,33,1))/np.sqrt(33)]
        if self.backend=='cuda':
            with self.assertRaisesRegex(ValueError,'<=32'):self.sampler(tensors)
        else:
            with self.sampler(tensors) as sampler:
                result=sampler.sample([[.3]],[[.5]])
                self.assertTrue(np.isfinite(result['selected_refined_probabilities']).all())

class TestNativeCPU(NativeCases,unittest.TestCase):backend='cpu'
class TestNativeCUDA(NativeCases,unittest.TestCase):backend='cuda'

if __name__=='__main__':unittest.main()
