import ctypes as C
import os
from pathlib import Path
import shutil
import unittest
import numpy as np
from gbslab.native_gaussian_reset import lossy_squeezed_components, DP
from gbslab.native_direct_click import NativeDirectClickSampler


@unittest.skipUnless(os.environ.get('GBSLAB_DIRECT_CLICK_LIBRARY'),
                     'explicit direct-click native library required')
class NativeDirectClickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=Path(os.environ['GBSLAB_DIRECT_CLICK_LIBRARY'])
        cls.runtime=[Path(shutil.which('g++')).parent]

    def model(self,**kwargs):
        return NativeDirectClickSampler(kwargs.pop('v',np.eye(2,dtype=complex)),
            kwargs.pop('x',np.zeros((4,4))),self.library,runtime_dirs=self.runtime,**kwargs)

    def test_vacuum_and_empty_batch(self):
        with self.model(workers=16) as model:
            result=model.sample(128,0)
            self.assertEqual(result['clicks'].dtype,np.uint8)
            self.assertFalse(result['clicks'].any())
            self.assertEqual(result['statistics']['detector_steps'],256)
            self.assertEqual(result['statistics']['zero_click_samples'],128)
            self.assertEqual(model.sample(0,0)['clicks'].shape,(0,2))
            self.assertEqual(model.sample(2,0,workers=256)['statistics']['workers'],2)

    def test_coherent_bernoulli_law(self):
        mean=np.array([1.,.7,-.2,.4]);p=-np.expm1(-(mean[:2]**2+mean[2:]**2)/2)
        with self.model(mean=mean,workers=8) as model:
            clicks=model.sample(100000,710001)['clicks']
        histogram=np.bincount(clicks@np.array([1,2]),minlength=4)/len(clicks)
        exact=np.array([(1-p[0])*(1-p[1]),p[0]*(1-p[1]),(1-p[0])*p[1],p[0]*p[1]])
        self.assertLess(.5*abs(histogram-exact).sum(),.012)

    def test_single_squeezed_mode_exact_binary_probability(self):
        r=.8;x=np.diag([.5*np.expm1(2*r),.5*np.expm1(-2*r)])
        with self.model(v=np.ones((1,1),complex),x=x,workers=8) as model:
            result=model.sample(100000,710002)
        p=1-1/np.cosh(r)
        self.assertLess(abs(result['clicks'].mean()-p),7*np.sqrt(p*(1-p)/100000))

    def test_schedule_replay_and_noise_copy(self):
        rng=np.random.default_rng(710003);v=np.linalg.qr(rng.normal(size=(8,3))+1j*rng.normal(size=(8,3)))[0]
        x,variance=lossy_squeezed_components(3,.9,.65);original=variance.copy()
        with self.model(v=v,x=x,noise_variance=variance) as model:
            variance[:]=0
            np.testing.assert_array_equal(original,model.noise_variance)
            a=model.sample(1001,710004,workers=1)
            for workers in (4,16):
                b=model.sample(1001,710004,workers=workers)
                np.testing.assert_array_equal(a['clicks'],b['clicks'])
                self.assertEqual(a['statistics']['clicks'],b['statistics']['clicks'])
                self.assertEqual(b['statistics']['completed_samples'],1001)

    def test_first_mode_matches_gaussian_vacuum_formula(self):
        rng=np.random.default_rng(710005);v=np.linalg.qr(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)))[0]
        x,variance=lossy_squeezed_components(2,.9,.65);o=np.block([[v.real,-v.imag],[v.imag,v.real]])
        sigma=np.eye(8)/2+o@(x+np.diag(variance))@o.T
        p=1-1/np.sqrt(np.linalg.det(sigma[np.ix_([0,4],[0,4])]+np.eye(2)/2))
        with self.model(v=v,x=x,noise_variance=variance,workers=8) as model:
            clicks=model.sample(100000,710006)['clicks']
        self.assertLess(abs(clicks[:,0].mean()-p),7*np.sqrt(p*(1-p)/len(clicks)))

    def test_invalid_arguments_and_lifetime(self):
        for workers in (0,-1,257,True,1.5):
            with self.assertRaises(ValueError):self.model(workers=workers)
        for kwargs in ({'x':-np.eye(4)},{'x':np.eye(3)},{'v':np.ones((2,2))},
                       {'noise_variance':[-1,0,0,0]},{'mean':[0,0,0,np.nan]}):
            with self.assertRaises(ValueError):self.model(**kwargs)
        with self.model() as model:
            for count,seed in ((-1,0),(True,0),(2**31,0),(1,-1),(1,2**64),(1,True)):
                with self.assertRaises(ValueError):model.sample(count,seed)
            with self.assertRaises(ValueError):model.sample(1,0,workers=0)
        model.close()
        with self.assertRaises(RuntimeError):model.sample(1,0)

    def test_native_failure_restores_precision_and_recovers(self):
        with self.model(workers=4) as model:
            model.lib.gr_precision.argtypes=[DP]
            before=np.zeros(5);after=np.zeros(5);model.lib.gr_precision(before.ctypes.data_as(DP))
            model.mean[0]=np.nan
            with self.assertRaises(ArithmeticError):model.sample(10,0)
            model.lib.gr_precision(after.ctypes.data_as(DP));self.assertEqual(before[0],after[0])
            model.mean[0]=0
            self.assertFalse(model.sample(10,0)['clicks'].any())


if __name__=='__main__':unittest.main()
