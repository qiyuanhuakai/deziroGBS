"""Explicit dyadic transport for the existing native forest event/scoring kernel.

Event masses, error indicators and log diagnostics decode to exact Fractions.
The final Bernoulli probability remains the guarded binary64 native result.
Loading is explicit; imports and unit tests never build a native library.
"""
import ctypes as ct
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from . import batched_forest as legacy
from .positive_forest import ForestCompletionSampler

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY=ROOT/'.cache/wire-forest-native/libwire_forest.dll'
BYTE=ct.POINTER(ct.c_ubyte)


def encode_dyadic(value):
    value=Fraction(value)
    if not value:return bytes(24)
    sign=int(value<0);value=abs(value);den=value.denominator
    if den&(den-1):raise ValueError('Wire input must be exactly dyadic')
    exponent=-(den.bit_length()-1);mantissa=value.numerator
    while not mantissa&1:mantissa>>=1;exponent+=1
    if mantissa.bit_length()>128 or not -(1<<31)<=exponent<(1<<31):raise ValueError('Wire scalar outside format')
    return bytes([1,sign,0,0])+exponent.to_bytes(4,'little',signed=True)+mantissa.to_bytes(16,'little')


def decode_dyadic(data):
    data=bytes(data)
    if len(data)!=24:raise ValueError('Wire scalar must have 24 bytes')
    if not any(data):return Fraction(0)
    if data[0]!=1 or data[1]>1 or data[2] or data[3]:raise ValueError('Noncanonical wire scalar')
    exponent=int.from_bytes(data[4:8],'little',signed=True);mantissa=int.from_bytes(data[8:],'little')
    if not mantissa&1 or not -20000<=exponent<=20000:raise ValueError('Invalid native dyadic scalar')
    numerator=-mantissa if data[1] else mantissa
    return Fraction(numerator<<exponent) if exponent>=0 else Fraction(numerator,1<<(-exponent))


class _Conditional(ct.Structure):
    _fields_=[('probability',ct.c_double),('precision_used',ct.c_int),('refinement_attempted',ct.c_int),('zero_score_reason',ct.c_int*2),('scalars',ct.c_ubyte*144),('first_error',ct.c_char*512)]


class WireForestKernel(legacy.BatchedForestKernel):
    def __init__(self,covariance,*,order=None,library=None):
        started=perf_counter();self._handle=None;self._closed_info=None;self._dll_directories=[]
        raw=np.asarray(covariance)
        if np.iscomplexobj(raw) and np.any(raw.imag!=0):raise ValueError('Covariance must be real')
        V=np.array(raw.real,dtype=np.float64,order='C',copy=True)
        if V.ndim!=2 or V.shape[0]==0 or V.shape[0]!=V.shape[1] or V.shape[0]%2 or not np.all(np.isfinite(V)) or not np.array_equal(V,V.T):raise ValueError('Covariance must be finite, even and exactly symmetric')
        self.m=V.shape[0]//2;self.order=legacy._order(order,self.m);self._positions={mode:j for j,mode in enumerate(self.order)};self.input_covariance_sha256=hashlib.sha256(V.tobytes()).hexdigest()
        self.library_path=Path(library or os.environ.get('GBSLAB_WIRE_FOREST_LIBRARY',DEFAULT_LIBRARY)).resolve()
        if not self.library_path.is_file():raise FileNotFoundError(f'Explicit wire forest library is missing: {self.library_path}')
        self.library_sha256=legacy._sha(self.library_path);metadata_path=self.library_path.parent/'build.json'
        if not metadata_path.is_file():raise RuntimeError('Wire library requires build provenance')
        self.build_metadata_sha256=legacy._sha(metadata_path);metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata['library_sha256']!=self.library_sha256:raise RuntimeError('Wire library differs from build metadata')
        for name,sha in metadata['source_hashes'].items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or not (ROOT/path).is_file() or legacy._sha(ROOT/path)!=sha:raise RuntimeError('Wire build source binding differs: '+name)
        if os.name=='nt':
            for dependency in metadata.get('runtime_dependencies',[]):
                path=Path(dependency['path'])
                if legacy._sha(path)!=dependency['sha256']:raise RuntimeError('Runtime dependency hash mismatch')
                self._dll_directories.append(os.add_dll_directory(str(path.parent)))
        self._wire=ct.CDLL(str(self.library_path));lib=self._wire
        signatures={
            'bfw_formats':([ct.POINTER(ct.c_int),ct.c_size_t],ct.c_int),
            'bfw_create':([ct.c_int,ct.POINTER(ct.c_double),legacy._INT_PTR,ct.c_char_p,ct.c_size_t],ct.c_void_p),
            'bfw_destroy':([ct.c_void_p],None),'bfw_reset':([ct.c_void_p],None),
            'bfw_diagnostics':([ct.c_void_p,ct.POINTER(legacy._Diagnostics)],ct.c_int),
            'bfw_query':([ct.c_void_p,ct.POINTER(legacy._Query),ct.c_int,ct.c_int,BYTE,BYTE,BYTE,ct.c_size_t,ct.c_char_p,ct.c_size_t],ct.c_int),
            'bfw_probability':([ct.c_void_p,ct.POINTER(legacy._Query),ct.c_int,BYTE,ct.c_size_t,BYTE,BYTE,BYTE,ct.c_size_t,ct.POINTER(_Conditional),ct.c_char_p,ct.c_size_t],ct.c_int),
            'scalar_wire_control_word':([],ct.c_int),'bfw_mxcsr':([],ct.c_uint)}
        for name,(args,result) in signatures.items():getattr(lib,name).argtypes=args;getattr(lib,name).restype=result
        values=(ct.c_int*10)();status=lib.bfw_formats(values,10)
        expected={0:2,1:24,6:ct.sizeof(legacy._Query),7:ct.sizeof(legacy._Diagnostics),8:ct.sizeof(_Conditional),9:ct.sizeof(ct.c_double)}
        if status or any(values[k]!=v for k,v in expected.items()):raise RuntimeError('Wire forest ABI mismatch')
        self.formats=dict(zip(('abi_version','wire_bytes','long_double_mantissa_bits','quad_mantissa_bits','maximum_modes','maximum_retained','query_bytes','diagnostics_bytes','conditional_bytes','double_bytes'),map(int,values)))
        self._lib=SimpleNamespace(bf_destroy=lib.bfw_destroy,bf_reset=lib.bfw_reset,bf_diagnostics=lib.bfw_diagnostics)
        order_buffer=np.asarray(self.order,np.int32);error=ct.create_string_buffer(512)
        self._handle=lib.bfw_create(self.m,V.ctypes.data_as(ct.POINTER(ct.c_double)),order_buffer.ctypes.data_as(legacy._INT_PTR),error,len(error))
        if not self._handle:raise ValueError(error.value.decode() or 'Native wire constructor failed')
        self.initialization_seconds=perf_counter()-started

    def _wire_request(self,zeros,retained,mode,omitted,edges):
        self._require_open();zeros=tuple(sorted(legacy._indices(zeros,self.m,'zeros'),key=self._positions.__getitem__));retained=legacy._indices(retained,self.m,'retained');omitted=legacy._indices(omitted,self.m,'omitted');mode=legacy._indices((mode,),self.m,'mode')[0];edges=tuple(tuple(e) for e in edges)
        for e in edges:
            if len(e)!=2:raise ValueError('Edges must have two endpoints')
            legacy._indices(e,self.m,'edge')
        buffers=[np.asarray(v,np.int32) for v in (zeros,retained,omitted,tuple(j for e in edges for j in e))];query=legacy._Query(*(b.ctypes.data_as(legacy._INT_PTR) for b in buffers),mode,len(zeros),len(retained),len(omitted),len(edges));count=1+2*(1+len(omitted)+len(edges))
        return buffers,query,(ct.c_ubyte*(24*count))(),(ct.c_ubyte*(24*count))(),np.zeros(count,np.uint8)

    @staticmethod
    def _decoded_batch(values,indicators,zeros,query,precision):
        raw=bytes(values);errors=bytes(indicators);count=len(zeros)
        values=np.array([decode_dyadic(raw[j*24:(j+1)*24]) for j in range(count)],dtype=object);indicators=np.array([decode_dyadic(errors[j*24:(j+1)*24]) for j in range(count)],dtype=object)
        batch=legacy.BatchedForestKernel._batch(values,indicators,zeros,query.omitted_count,query.edge_count,precision)
        batch.update(values_wire=raw,indicators_wire=errors,scalar_transport='canonical dyadic bytes, decoded exactly as Fraction')
        return batch

    def query(self,zeros,retained,mode,omitted=(),edges=(),*,precision='longdouble',force_rebuild=False):
        if precision not in ('longdouble','quad'):raise ValueError('Raw query requires explicit precision')
        buffers,q,v,e,z=self._wire_request(zeros,retained,mode,omitted,edges);error=ct.create_string_buffer(512)
        status=self._wire.bfw_query(self._handle,ct.byref(q),legacy.PRECISIONS[precision],bool(force_rebuild),v,e,z.ctypes.data_as(BYTE),len(z),error,len(error));legacy._raise_status(status,error)
        return self._decoded_batch(v,e,z,q,precision)

    def probability(self,zeros,retained,mode,omitted=(),edges=(),*,precision='auto',max_relative_roundoff=1e-6):
        if precision not in legacy.PRECISIONS:raise ValueError('Invalid precision policy')
        guard=Fraction(max_relative_roundoff)
        if not 0<guard<1:raise ValueError('Relative guard must be in (0,1)')
        wire=(ct.c_ubyte*24).from_buffer_copy(encode_dyadic(guard));buffers,q,v,e,z=self._wire_request(zeros,retained,mode,omitted,edges);result=_Conditional();error=ct.create_string_buffer(512)
        status=self._wire.bfw_probability(self._handle,ct.byref(q),legacy.PRECISIONS[precision],wire,24,v,e,z.ctypes.data_as(BYTE),len(z),ct.byref(result),error,len(error));legacy._raise_status(status,error)
        actual=legacy.PRECISION_NAMES[result.precision_used];batch=self._decoded_batch(v,e,z,q,actual);raw=bytes(result.scalars);numbers=tuple(decode_dyadic(raw[j*24:(j+1)*24]) for j in range(6));reasons=tuple(legacy.ZERO_REASONS[j] for j in result.zero_score_reason)
        batch.update(click_probability=float(result.probability),precision_requested=precision,refinement_attempted=bool(result.refinement_attempted),first_error=bytes(result.first_error).decode(),log_scores=tuple(None if reasons[j] else numbers[j] for j in range(2)),log_score_indicators=numbers[2:4],zero_score_reasons=reasons,maximum_passed_event_relative_indicator=numbers[4],maximum_passed_log_score_indicator=numbers[5],conditional_scalars_wire=raw)
        return batch

    def cache_info(self):
        info=super().cache_info();info.update(kind='wire_shared_schur_batched_forest_native',scalar_transport='24-byte canonical dyadic; exact Fraction decoding',arithmetic_environment='x87 extended precision / nearest; SSE nearest with FTZ and DAZ off; control fields restored')
        return info


class WireForestSampler(legacy.BatchedForestSampler):
    def __init__(self,model,max_clicks,*,forest='chain',selection='click_influence',order=None,cache_size=8192,max_relative_roundoff=1e-6,precision='auto',library=None):
        started=perf_counter()
        if precision not in legacy.PRECISIONS:raise ValueError('Invalid precision')
        if not hasattr(model,'V'):raise ValueError('Original Gaussian covariance required')
        ForestCompletionSampler.__init__(self,model,max_clicks,forest=forest,selection=selection,order=order,cache_size=cache_size,max_relative_roundoff=max_relative_roundoff)
        self.precision=precision;legacy.install_canonical_selector(self);self.kernel=WireForestKernel(model.V,order=self.order,library=library)
        if self.max_clicks>self.kernel.formats['maximum_retained']:self.kernel.close();raise ValueError('Retained budget exceeds native cap')
        self._event_backend=self.kernel;self.oracle_kind='native_wire_shared_schur_batch_with_common_PZ_scale';self.diagnostics['initialization_seconds']=perf_counter()-started

    def cache_info(self):
        info=super().cache_info();info['scoring_backend']='unchanged bf_probability native kernel via explicit wire adapter';return info
