import os,shutil,unittest,json,ctypes as C,math
from pathlib import Path
import numpy as np
from gbslab.native_projected_pnr import NativeProjectedPNRSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_PROJECTED_PNR_LIBRARY'),'explicit projected PNR library required')
class ProjectedPNRTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativeProjectedPNRSampler(kwargs.pop('v',np.ones((1,1),complex)),kwargs.pop('t',[1/3]),os.environ['GBSLAB_PROJECTED_PNR_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)
    def test_archived_worker_replay(self):
        d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');base=ROOT/'results/iteration76_projected_pilot_v3';row=json.loads((base/'row_1_31_projected.json').read_text());archive=np.load(base/'counts_1_31_projected.npz')['counts']
        with self.model(v=d['v'],t=d['t'],retained=31) as model:
            a=model.sample(1024,row['seed'],workers=1);b=model.sample(1024,row['seed'],workers=16)
        np.testing.assert_array_equal(a['counts'],archive[:1024]);np.testing.assert_array_equal(a['counts'],b['counts']);self.assertEqual(a['statistics']['photons'],int(a['counts'].sum()))
    def test_same_random_prefix_coupling(self):
        d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');values=[]
        for k in (7,31):
            with self.model(v=d['v'],t=d['t'],retained=k,workers=16) as model:values.append(model.sample(2048,760901)['counts'])
        reference=values[1];mask=reference.sum(axis=1)<=7;np.testing.assert_array_equal(values[0][mask],reference[mask]);self.assertTrue(np.all(values[0][np.any(values[0]!=reference,axis=1)].sum(axis=1)>=8))
    def test_128_input_capacity_and_pure_parity(self):
        with self.model(v=np.eye(128,dtype=complex),t=np.full(128,.05),retained=7) as model:
            a=model.sample(128,760902,workers=1)['counts'];b=model.sample(128,760902,workers=16)['counts']
        np.testing.assert_array_equal(a,b);total=a.sum(axis=1);self.assertTrue(np.all(total[total<=7]%2==0));self.assertEqual(a.dtype,np.uint32)
    def test_repeated_factor_analytic_moment_in_128_inputs(self):
        n=128;t=np.full(n,.3);t[1::2]*=-1;DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);ids=np.zeros(4,np.int32);mean=np.zeros(2*n)
        with self.model(v=np.eye(n,dtype=complex),t=t,retained=7) as model:
            fn=model.lib.prefix76_probe;fn.argtypes=[C.c_void_p,DP,DP,DP,C.c_int,IP,C.c_double,DP,DP,DP,DP,DP,C.c_char_p,C.c_int]
            for tau in (0.,.2,1.):
                mass=np.zeros(1);derivative=np.zeros(1);cov=np.zeros((2*n,2*n));outmean=np.zeros(2*n);diag=np.zeros(3);error=C.create_string_buffer(4096);ptr=lambda x:x.ctypes.data_as(DP);rc=fn(model.handle,ptr(t),ptr(mean),ptr(mean),4,ids.ctypes.data_as(IP),tau,ptr(mass),ptr(derivative),ptr(cov),ptr(outmean),ptr(diag),error,len(error));self.assertEqual(rc,0,error.value)
                tj=t[0];den=1-tau*tau*tj*tj;M=-tj/den;N=tau*tj*tj/den;Md=2*tau*tj**3/den**2;Nd=-tj*tj*(1+tau*tau*tj*tj)/den**2;value=prime=0.
                for pairs in range(3):
                    q=4-2*pairs;c=math.factorial(4)**2/(math.factorial(q)*(2**pairs*math.factorial(pairs))**2);value+=c*M**(2*pairs)*N**q
                    if pairs:prime+=c*2*pairs*M**(2*pairs-1)*N**q*Md
                    if q:prime+=c*q*M**(2*pairs)*N**(q-1)*Nd
                self.assertAlmostEqual(mass[0],value,places=11);self.assertAlmostEqual(derivative[0],prime,places=10);self.assertLessEqual(diag[0],4);self.assertLess(diag[2],1e-13)
    def test_vacuum_empty_and_coherent(self):
        with self.model(t=[0.],retained=7) as model:
            self.assertFalse(model.sample(64,0)['counts'].any());self.assertEqual(model.sample(0,0)['counts'].shape,(0,1))
        with self.model(t=[0.],retained=3,mean=[2.,0.],workers=16) as model:x=model.sample(30000,760903)['counts'][:,0]
        self.assertLess(abs(x.mean()-2),.05);self.assertLess(abs(x.var()-2),.09)
    def test_invalid_resource_limit_and_no_photon_cutoff(self):
        for value in (True,-1,32,1.5):
            with self.assertRaises(ValueError):self.model(retained=value)
        with self.model(v=np.eye(128,dtype=complex),t=np.full(128,.2),retained=31) as model:
            with self.assertRaisesRegex(ArithmeticError,'resource estimate'):model.sample(1,0)
        d=np.load(ROOT/'results/iteration76_projected_pilot_v3/counts_1_3_projected.npz')['counts'];self.assertGreater(int(d.sum(axis=1).max()),4)
