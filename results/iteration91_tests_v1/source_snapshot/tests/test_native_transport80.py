import os,shutil,unittest,ctypes as C
from pathlib import Path
import numpy as np
from gbslab.native_cached_transport_pnr import NativeCachedTransportPNRSampler
from gbslab.native_transport_number_pnr import NativeTransportNumberPNRSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_CACHED_TRANSPORT_LIBRARY'),'explicit cached transport library required')
class Transport80Tests(unittest.TestCase):
    def model(self,case='strong4',**kw):
        d=np.load(ROOT/f'results/iteration80_ipf_pilot_v1/{case}.npz')
        return NativeCachedTransportPNRSampler(d['v'],kw.pop('t',.5),os.environ['GBSLAB_CACHED_TRANSPORT_LIBRARY'],couplings=kw.pop('couplings',d['couplings']),latent_weights=d['weights'],runtime_dirs=[Path(shutil.which('g++')).parent],**kw)
    def test_strict_archive_replay(self):
        t=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz')['t']
        with self.model(t=t,exact_total_budget=64) as model:r=model.sample(1024,0x88000000000000+2**40)
        np.testing.assert_array_equal(r['counts'],np.load(ROOT/'results/iteration77_number_pilot_v2/counts_1.npz')['counts'][:1024])
    def test_cached_matches_integer_export_archive(self):
        with self.model(exact_total_budget=0) as model:r=model.sample(1024,0xb0000000000000,fixed_total=14)
        np.testing.assert_array_equal(r['counts'],np.load(ROOT/'results/iteration80_alias_audit_v2/strong4.npz')['counts'])
    def test_workers_preserve_all_counts(self):
        with self.model(exact_total_budget=0) as model:a=model.sample(1024,80801,workers=1,fixed_total=34);b=model.sample(1024,80801,workers=16,fixed_total=34)
        np.testing.assert_array_equal(a['counts'],b['counts']);self.assertTrue(np.all(a['counts'].sum(axis=1)==34))
    def test_identity_preserves_even_each_port(self):
        with self.model('identity2',exact_total_budget=0) as model:r=model.sample(1024,80802,fixed_total=14)
        self.assertFalse((r['counts']%2).any())
    def test_balanced_phase_pairs(self):
        with self.model('balanced_phase2',exact_total_budget=0) as model:r=model.sample(1024,80803,fixed_total=14)
        np.testing.assert_array_equal(r['counts'],np.full((1024,2),7))
    def test_empty_vacuum_and_lifetime(self):
        model=self.model(t=0.)
        self.assertEqual(model.sample(0,0)['counts'].shape,(0,4));self.assertFalse(model.sample(32,0)['counts'].any());model.close();model.close()
        with self.assertRaises(RuntimeError):model.sample(1,0)
    def test_invalid_totals_and_budget(self):
        with self.model() as model:
            for total in (1,3,-2,2.0,True):
                with self.assertRaises(ValueError):model.sample(1,0,fixed_total=total)
        for B in (1,-2,True,500):
            with self.assertRaises(ValueError):self.model(exact_total_budget=B)
    def test_bad_coupling_rejected(self):
        d=np.load(ROOT/'results/iteration80_ipf_pilot_v1/strong4.npz');p=d['couplings'].copy();p[0,0,0]=-1
        with self.assertRaises(ValueError):self.model(couplings=p)
        p=np.full_like(p,1/16)
        with self.assertRaises(ValueError):self.model(couplings=p)
    def test_readonly_tables_and_error_scope(self):
        with self.model() as model:
            self.assertLess(model.pair_constraint_error,1e-8);self.assertLess(model.cross_pair_constraint_error,1e-8);self.assertGreater(model.alias_payload_bytes,0)
            with self.assertRaises(ValueError):model.actual_pair_masses[0,0]=0
    def test_old_cdf_tiny_interval_failure_retained(self):
        d=np.load(ROOT/'results/iteration80_ipf_pilot_v1/identity2.npz')
        with NativeTransportNumberPNRSampler(d['v'],.5,ROOT/'results/iteration80_transport_pilot_v2/transport.dll',couplings=d['couplings'],latent_weights=d['weights'],exact_total_budget=0,runtime_dirs=[Path(shutil.which('g++')).parent]) as model:
            with self.assertRaisesRegex(ArithmeticError,'interval'):model.sample(1,0,fixed_total=14)

@unittest.skipUnless(os.environ.get('GBSLAB_ORDER80_LIBRARY'),'explicit order library required')
class Order80Tests(unittest.TestCase):
    def call(self,cov):
        self.directory=os.add_dll_directory(str(Path(shutil.which('g++')).parent));lib=C.CDLL(os.environ['GBSLAB_ORDER80_LIBRARY']);fn=lib.order80_build;D=C.POINTER(C.c_double);I=C.POINTER(C.c_int32);fn.argtypes=[C.c_int,D,I,D,D,C.c_char_p,C.c_int];fn.restype=C.c_int;m=len(cov)//2;cov=np.ascontiguousarray(cov);orders=np.zeros((6,m),np.int32);vac=np.zeros(m);corr=np.zeros((m,m));err=C.create_string_buffer(4096);rc=fn(m,cov.ctypes.data_as(D),orders.ctypes.data_as(I),vac.ctypes.data_as(D),corr.ctypes.data_as(D),err,len(err));return rc,orders,vac,corr,err.value
    def tearDown(self):
        if hasattr(self,'directory'):self.directory.close()
    def test_thermal_analytic_probabilities(self):
        occupation=np.array([.1,.5,1.,2.]);rc,orders,vac,corr,error=self.call(np.diag(np.tile(1+2*occupation,2)));self.assertEqual(rc,0,error);np.testing.assert_allclose(vac,1/(1+occupation),atol=1e-15);np.testing.assert_allclose(corr,np.diag(vac*(1-vac)),atol=1e-15)
        for order in orders:np.testing.assert_array_equal(np.sort(order),np.arange(4))
        np.testing.assert_array_equal(orders[1],np.arange(4));np.testing.assert_array_equal(orders[2],np.arange(3,-1,-1))
    def test_nonsymmetric_rejected(self):
        cov=np.eye(8);cov[0,1]=.1;rc,*_=self.call(cov);self.assertNotEqual(rc,0)
    def test_invalid_covariance_rejected(self):
        rc,*_=self.call(-2*np.eye(8));self.assertNotEqual(rc,0)
