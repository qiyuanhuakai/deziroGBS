import ctypes as C
import os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_structured_pnr import NativeStructuredPNRSampler,DP
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_STRUCTURED_PNR_LIBRARY'),'explicit structured PNR library required')
class StructuredPNRTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativeStructuredPNRSampler(kwargs.pop('v',np.ones((1,1),complex)),kwargs.pop('t',[1/3]),os.environ['GBSLAB_STRUCTURED_PNR_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)
    def test_old_failure_seed_now_completes(self):
        for k in (1,2,3):
            with self.model(retained=k) as model:r=model.sample(100000,736001,workers=16)
            self.assertEqual(r['statistics']['completed_samples'],100000)
            self.assertEqual(r['statistics']['photons'],int(r['counts'].sum()))
            # Completion is numerical support, not an exact-parity claim.
            self.assertGreater(np.count_nonzero(r['counts']%2),0)
    def test_exact_odd_endpoint_zeros(self):
        with self.model() as model:
            fn=model.lib.pnrs74_endpoint;fn.argtypes=[C.c_void_p,DP,DP,C.c_int,C.POINTER(C.c_int),C.c_double,DP,C.c_char_p,C.c_int]
            for k in (1,3,5,7):
                ids=np.zeros(k,np.int32);w=np.zeros(2);out=np.ones(1);error=C.create_string_buffer(4096)
                rc=fn(model.handle,model.squeezing.ctypes.data_as(DP),w.ctypes.data_as(DP),k,ids.ctypes.data_as(C.POINTER(C.c_int)),0.,out.ctypes.data_as(DP),error,len(error))
                self.assertEqual(rc,0,error.value);self.assertEqual(out[0],0)
    def test_noisy_archived_worker_replay(self):
        data=np.load(ROOT/'results/iteration74_structured_pilot_v1/counts_3_3.npz')
        with self.model(v=data['v'],t=data['tanh_squeezing'],mean=data['mean'],noise_variance=data['variance'],retained=3) as model:
            a=model.sample(2048,742401,workers=1);b=model.sample(2048,742401,workers=16)
        np.testing.assert_array_equal(a['counts'],data['counts'][:2048]);np.testing.assert_array_equal(a['counts'],b['counts'])
    def test_vacuum_empty_and_closed(self):
        with self.model(t=[0.]) as model:
            self.assertFalse(model.sample(200,0)['counts'].any());self.assertEqual(model.sample(0,0)['counts'].shape,(0,1))
        with self.assertRaises(RuntimeError):model.sample(1,0)
    def test_invalid_parameters_and_recovery(self):
        for t in ([1.],[-1.],[np.nan],[[.1]]):
            with self.assertRaises(ValueError):self.model(t=t)
        with self.model() as model:
            for count,seed in ((True,0),(-1,0),(1,-1),(1,2**64)):
                with self.assertRaises(ValueError):model.sample(count,seed)
            model.squeezing[0]=1
            with self.assertRaises(ArithmeticError):model.sample(1,0)
            model.squeezing[0]=1/3;self.assertEqual(model.sample(128,736001)['counts'].shape,(128,1))
