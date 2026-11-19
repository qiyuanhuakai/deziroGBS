import unittest
from unittest.mock import patch
import numpy as np
from gbslab.gaussian import local_unitary
from gbslab.local_mps import local_circuit, build_local_core_layerwise, sample_local_threshold
from gbslab.local_displacement import LocalPropagator, iter_local_displacements


class LocalDisplacementTests(unittest.TestCase):
    def test_matrix_free_generation_stream_and_no_identity(self):
        expected,u=local_circuit(7,5,np.random.default_rng(1401))
        with patch('gbslab.local_mps.np.eye',side_effect=AssertionError('dense identity')):
            actual,none=local_circuit(7,5,np.random.default_rng(1401),materialize_unitary=False)
        self.assertIsNone(none)
        for a,b in zip(actual,expected):
            self.assertEqual(a[0],b[0])
            for x,y in zip(a[1:],b[1:]): np.testing.assert_array_equal(x,y)
        np.testing.assert_allclose(u,local_unitary(7,5,np.random.default_rng(1401)),atol=1e-15)

    def test_complex_propagation_odd_even_empty_depth(self):
        for m,depth in ((2,0),(3,1),(8,4),(9,12)):
            gates,u=local_circuit(m,depth,np.random.default_rng(1402))
            p=LocalPropagator(m,gates); rng=np.random.default_rng(1403)
            x=rng.normal(size=(11,m))+1j*rng.normal(size=(11,m)); original=x.copy()
            y=p.apply(x)
            np.testing.assert_allclose(y,x@u.T,atol=8e-15,rtol=1e-14)
            np.testing.assert_allclose(np.sum(abs(y)**2,axis=1),np.sum(abs(x)**2,axis=1),atol=1e-12)
            np.testing.assert_array_equal(x,original)
            self.assertEqual(p.apply(np.empty((0,m))).shape,(0,m))

    def test_real_noise_has_correct_noncircular_moments(self):
        gates,u=local_circuit(5,4,np.random.default_rng(1404)); p=LocalPropagator(5,gates)
        variance=.7
        transfer=.5j*np.sqrt(variance)*p.apply(np.eye(5)).T
        np.testing.assert_allclose(transfer@transfer.conj().T,variance/4*np.eye(5),atol=1e-15)
        np.testing.assert_allclose(transfer@transfer.T,-variance/4*(u@u.T),atol=1e-15)
        self.assertGreater(np.linalg.norm(transfer@transfer.T),.1)

    def test_stream_batches_preserve_normals(self):
        gates,u=local_circuit(7,4,np.random.default_rng(1405)); p=LocalPropagator(7,gates)
        reference=.5j*np.sqrt(.8)*(np.random.default_rng(1406).normal(size=(31,7))@u.T)
        for batch in (1,7,32):
            result=np.concatenate(list(iter_local_displacements(p,.8,31,np.random.default_rng(1406),batch_size=batch)))
            np.testing.assert_allclose(result,reference,atol=2e-15)

    def test_batch_invariance_of_complete_sampling(self):
        gates,_=local_circuit(4,4,np.random.default_rng(1407),materialize_unitary=False)
        p=LocalPropagator(4,gates); tensors,_=build_local_core_layerwise(4,.3,4,4,gates)
        results=[]
        for batch in (1,7,32):
            noise=np.random.default_rng(1408); measurement=np.random.default_rng(1409)
            results.append(np.concatenate([sample_local_threshold(tensors,alpha,measurement)[0]
                for alpha in iter_local_displacements(p,.8,31,noise,batch_size=batch)]))
        for x in results[1:]: np.testing.assert_array_equal(results[0],x)

    def test_zero_samples_no_draws_and_zero_variance(self):
        p=LocalPropagator(2,[]); a=np.random.default_rng(1410); b=np.random.default_rng(1410)
        self.assertEqual(list(iter_local_displacements(p,1,0,a)),[])
        self.assertEqual(a.random(),b.random())
        np.testing.assert_array_equal(next(iter_local_displacements(p,0,3,a)),np.zeros((3,2)))

    def test_invalid_inputs(self):
        for gates in ([('phases',[0])],[('pair',0,np.ones((2,2)))],
            [('pair',0,np.eye(2)),('pair',1,np.eye(2))],[('unknown',)]):
            with self.assertRaises(ValueError): LocalPropagator(3,gates)
        p=LocalPropagator(3,[])
        for x in (np.ones((2,2)),np.full((2,3),np.nan)):
            with self.assertRaises(ValueError): p.apply(x)
        for var,count,batch in ((-1,2,1),(1,-1,1),(1,2,0)):
            with self.assertRaises(ValueError): list(iter_local_displacements(p,var,count,np.random.default_rng(1),batch_size=batch))


if __name__=='__main__': unittest.main()
