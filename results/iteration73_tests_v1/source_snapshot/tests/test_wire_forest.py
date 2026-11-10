"""Transport checks plus the inherited, substantive native numerical suite."""
import ctypes as ct
from fractions import Fraction
import hashlib,os
from pathlib import Path
import unittest
import numpy as np
import test_batched_forest_numerics as prior
from gbslab import batched_forest as legacy
from gbslab.wire_forest import WireForestKernel,WireForestSampler,encode_dyadic,decode_dyadic,DEFAULT_LIBRARY,BYTE,_Conditional
from gbslab.exact import NumericalProbabilityError,ThresholdModel


def load_library(cls):
    explicit=os.environ.get('GBSLAB_WIRE_FOREST_LIBRARY');cls.library=Path(explicit or DEFAULT_LIBRARY).resolve()
    if not cls.library.is_file():
        if explicit:raise FileNotFoundError(cls.library)
        raise unittest.SkipTest('Build the explicit scalar-wire forest library first')
    cls.library_sha256=hashlib.sha256(cls.library.read_bytes()).hexdigest()


class WireCodecTests(unittest.TestCase):
    def test_exact_dyadic_extremes_and_extra_bits(self):
        for value in (Fraction(0),Fraction(1,1<<16445),Fraction((1<<64)-1,1<<63),Fraction(-((1<<113)-1),1<<16494),Fraction(3<<16000)):
            self.assertEqual(decode_dyadic(encode_dyadic(value)),value)
        value=Fraction((1<<63)+1,1<<63);self.assertEqual(float(value),1.);self.assertNotEqual(value,Fraction(1))

    def test_noncanonical_and_non_dyadic_rejection(self):
        for value in (Fraction(1,3),Fraction((1<<129)-1)):
            with self.assertRaises(ValueError):encode_dyadic(value)
        for data in (bytes(23),bytes([0,1])+bytes(22),bytes([1,0,0,0])+bytes(4)+(2).to_bytes(16,'little'),bytes([1,2])+bytes(22)):
            with self.assertRaises(ValueError):decode_dyadic(data)


class WireForestTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):load_library(cls)

    def kernel(self,V=None,**kwargs):
        k=WireForestKernel(1.5*np.eye(8) if V is None else V,library=self.library,**kwargs);self.addCleanup(k.close);return k

    def test_actual_events_preserve_native_bits_and_readonly_storage(self):
        k=self.kernel();batch=k.query((),(),3,(0,1,2),((0,1),(1,2)),precision='quad')
        self.assertTrue(any(v!=Fraction(float(v)) for v in batch['values']))
        for name in ('values','indicators'):
            for i,value in enumerate(batch[name]):self.assertEqual(encode_dyadic(value),batch[name+'_wire'][24*i:24*(i+1)])
            self.assertFalse(batch[name].flags.writeable)
        self.assertTrue(all(v>=0 for v in batch['indicators']))
        self.assertAlmostEqual(float(batch['branches'][1].value),.2,delta=1e-16)

    def test_analytic_zero_flags_survive_transport(self):
        k=self.kernel(np.eye(8));b=k.probability((),(),0)
        self.assertEqual(b['click_probability'],0);self.assertTrue(b['branches'][1].exact_zero);self.assertEqual(b['branches'][1].value,0)
        self.assertEqual(b['zero_score_reasons'][1],'analytic_zero_baseline_branch')
        self.assertEqual(b['log_scores'][1],None)

    def test_control_fields_restored_on_success_and_native_failure(self):
        k=self.kernel();before=(k._wire.scalar_wire_control_word(),k._wire.bfw_mxcsr()&0xffc0)
        k.probability((),(),3,(0,1,2),((0,1),(1,2)))
        self.assertEqual(before,(k._wire.scalar_wire_control_word(),k._wire.bfw_mxcsr()&0xffc0))
        difficult=self.kernel((1+2**-40)*np.eye(16));before=(difficult._wire.scalar_wire_control_word(),difficult._wire.bfw_mxcsr()&0xffc0)
        with self.assertRaises(NumericalProbabilityError):difficult.probability((),(3,4,5,6),7,(0,1,2),((0,1),(1,2)))
        self.assertEqual(before,(difficult._wire.scalar_wire_control_word(),difficult._wire.bfw_mxcsr()&0xffc0))

    def test_malformed_guard_is_argument_error_without_refinement(self):
        k=self.kernel();buffers,q,v,e,z=k._wire_request((),(),3,(),());guard=(ct.c_ubyte*24).from_buffer_copy(bytes([2])+bytes(23));result=_Conditional();error=ct.create_string_buffer(512);before=k.cache_info()
        status=k._wire.bfw_probability(k._handle,ct.byref(q),2,guard,24,v,e,z.ctypes.data_as(BYTE),len(z),ct.byref(result),error,len(error))
        self.assertEqual(status,1);self.assertIn(b'guard',error.value);after=k.cache_info();self.assertEqual(after['auto_refinements'],before['auto_refinements']);self.assertEqual(after['probability_calls'],before['probability_calls'])
        with self.assertRaises(ValueError):k.probability((),(),3,max_relative_roundoff=Fraction((1<<80)-1,1<<81))

    def test_legacy_abi_still_refuses_incompatible_layout(self):
        if np.dtype(np.longdouble).itemsize==16:self.skipTest('Only relevant to the Windows 8-byte Python format')
        k=self.kernel() # Keep checked dependency directories registered.
        with self.assertRaisesRegex(RuntimeError,'format mismatch'):legacy.BatchedForestKernel(1.5*np.eye(8),library=self.library)

    def test_closed_handle_and_queries_are_independent(self):
        k=self.kernel();first=k.query((),(),3);saved=first['values'].copy();k.query((0,),(),3);np.testing.assert_array_equal(first['values'],saved);k.close();self.assertTrue(k.cache_info()['closed'])
        with self.assertRaises(RuntimeError):k.query((),(),3)

    def test_exact_guard_input_and_diagnostics_bytes(self):
        k=self.kernel();guard=Fraction(1,1<<19);b=k.probability((),(),3,max_relative_roundoff=guard);raw=b['conditional_scalars_wire'];numbers=[decode_dyadic(raw[i*24:(i+1)*24]) for i in range(6)]
        self.assertEqual(tuple(numbers[:2]),b['log_scores']);self.assertEqual(tuple(numbers[2:4]),b['log_score_indicators']);self.assertEqual(numbers[4],b['maximum_passed_event_relative_indicator']);self.assertLess(numbers[4],guard)


class WireForestNumericsTests(prior.BatchedForestNumericsTests):
    """All inherited refinement, failure-mass, endpoint, history and RNG tests."""
    @classmethod
    def setUpClass(cls):
        load_library(cls);cls.Sampler=WireForestSampler;cls.Kernel=WireForestKernel


if __name__=='__main__':unittest.main()
