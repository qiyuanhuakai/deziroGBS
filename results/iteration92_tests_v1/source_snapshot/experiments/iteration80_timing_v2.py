"""Fresh three-rotation timings, with calibration and constructor charged.
Identical input, count and thread policy; quality is NOT matched across methods.
File I/O is outside timing. Cached batches separately disclose initial setup.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,time,ctypes as C,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_cached_transport_pnr import NativeCachedTransportPNRSampler
from gbslab.native_wide_transport_alias_pnr import NativeWideTransportAliasPNRSampler
from gbslab.native_sphere_number_pnr import NativeSphereNumberPNRSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
from gbslab.native_alias_click import NativeAliasClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
dirs=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent];handles=[os.add_dll_directory(str(p)) for p in dirs]
bins={k:ROOT/v for k,v in dict(cached='results/iteration80_cached_pilot_v1/cached.dll',uncached='results/iteration80_wide_pilot_v1/transport.dll',sphere='results/iteration78_pilot_v2/sphere.dll',ipf='results/iteration80_wide_pilot_v1/ipf.dll',order='results/iteration80_order_pilot_v1/order.dll',cpu='results/iteration72_cuda_pilot_v1/cpu.dll',gpu='results/iteration74_fused_pilot_v2/philox.dll',alias='results/iteration78_alias_pilot_v1/table.dll',fgpu='results/iteration75_frugal_gpu_build_v3/frugal_fp64.dll',frugal='results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll').items()}
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int32);ptr=lambda a:a.ctypes.data_as(DP)
libs={k:C.CDLL(str(bins[k])) for k in ('ipf','order','frugal','fgpu')}
ipf=libs['ipf'].pair80_ipf;ipf.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_int,C.c_double,DP,DP,C.c_char_p,C.c_int];ipf.restype=C.c_int
orderfn=libs['order'].order80_build;orderfn.argtypes=[C.c_int,DP,IP,DP,DP,C.c_char_p,C.c_int];orderfn.restype=C.c_int
frugal=libs['frugal'].fc73_sample;frugal.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,DP,C.POINTER(C.c_uint8),DP,C.c_char_p,C.c_int];frugal.restype=C.c_int
fgpu=libs['fgpu'].fg72_sample;fgpu.argtypes=frugal.argtypes;fgpu.restype=C.c_int
cases=[('strong4',65536,[('cached',8),('uncached',8),('sphere',8),('gpu',8),('alias',0),('frugal',4),('fgpu',4)]),('eight',8192,[('gpu',4),('gpu',8),('neighbor',8),('alias',0),('frugal',5),('fgpu',5)]),('wide144',256,[('cached',8),('uncached',8),('sphere',8),('gpu',4),('neighbor',8),('frugal',2),('frugal',3),('fgpu',2),('fgpu',3)])]
save('plan.json',dict(scope=__doc__,cases=cases,repetitions=3,workers=16,alias_workers=1,seed_base=0xb3000000000000,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*bins.values()]}));rows=[];batchrows=[]
for case,(name,count,methods) in enumerate(cases):
    if name=='strong4':
        d=np.load(ROOT/'results/iteration80_ipf_pilot_v1/strong4.npz');v=d['v'];t=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz')['t'];weights=d['weights'];marginal=d['couplings'].sum(axis=2)
    elif name=='wide144':
        d=np.load(ROOT/'results/iteration80_wide_pilot_v1/input_2.npz');v=d['v'];t=np.full(v.shape[1],np.tanh(1.1));weights=d['weights'];marginal=d['marginals']
    else:
        d=np.load(ROOT/'results/iteration79_pilot_v1/click_input_1.npz');v=d['v'];t=None
    m,n=v.shape;x=d['x'] if t is None else np.diag(np.r_[-t/(1+t),t/(1-t)]);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@x@o.T);Q=np.ascontiguousarray(abs(v@v.T)**2/n)
    np.savez_compressed(OUT/f'input_{case}.npz',v=v,x=x,cov=cov,**({} if t is None else dict(t=t)))
    def make_transport(kind):
        table=np.zeros((len(weights),m,m));stats=np.zeros(4);error=C.create_string_buffer(4096);start=time.perf_counter();limit=min(256,2000000000//table.size)
        rc=ipf(m,len(weights),ptr(np.ascontiguousarray(weights)),ptr(np.ascontiguousarray(marginal)),ptr(Q),limit,1e-10,ptr(table),ptr(stats),error,len(error));assert rc==0,error.value
        calibrated=time.perf_counter()-start;cls=NativeCachedTransportPNRSampler if kind=='cached' else NativeWideTransportAliasPNRSampler
        model=cls(v,t,bins[kind],couplings=table,latent_weights=weights,exact_total_budget=8,workers=16,runtime_dirs=dirs)
        return model,dict(calibration_seconds=calibrated,calibration_iterations=int(stats[0]),calibration_residual=float(stats[2]),alias_payload_bytes=getattr(model,'alias_payload_bytes',None))
    def run(kind,budget,rep):
        seed=0xb3000000000000+(case*16+rep)*2**40;start=time.perf_counter();meta={};counts=None
        if kind in ('cached','uncached','sphere'):
            model,meta=make_transport(kind) if kind!='sphere' else (NativeSphereNumberPNRSampler(v,t,bins['sphere'],exact_total_budget=budget,workers=16,runtime_dirs=dirs),{})
            setup=time.perf_counter()-start
            with model:r=model.sample(count,seed)
            counts=r['counts'];bits=(counts>0).astype(np.uint8);meta.update(setup_seconds=setup,sampling_seconds=r['wall_seconds'],statistics=r['statistics'])
        elif kind in ('gpu','neighbor'):
            order=np.arange(m)
            if kind=='neighbor':
                orders=np.zeros((6,m),np.int32);vac=np.zeros(m);corr=np.zeros((m,m));err=C.create_string_buffer(4096);rc=orderfn(m,ptr(cov),orders.ctypes.data_as(IP),ptr(vac),ptr(corr),err,len(err));assert rc==0,err.value;order=orders[4].copy();meta['ordering_seconds']=time.perf_counter()-start
            with NativePhiloxClickSampler(v[order],x,bins['cpu'],gpu_library=bins['gpu'],branch_budget=budget,runtime_dirs=dirs) as model:r=model.sample(count,seed)
            bits=np.zeros_like(r['clicks']);bits[:,order]=r['clicks'];meta.update(cuda_timings=r['cuda_timings'],statistics=r['statistics'])
        elif kind=='alias':
            with NativeAliasClickSampler(cov,bins['alias'],workers=1,runtime_dirs=dirs) as model:r=model.sample(count,seed)
            bits=r['clicks'];meta['sampling_seconds']=r['wall_seconds']
        else:
            bits=np.zeros((count,m),np.uint8);times=np.zeros(6 if kind=='fgpu' else 7);err=C.create_string_buffer(4096);rc=(fgpu if kind=='fgpu' else frugal)(m,budget,count,801101+case*16+rep,16,ptr(cov),bits.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(times),err,len(err));assert rc==0,err.value;meta['component_timings']=times.tolist()
        return bits,counts,dict(wall_seconds=time.perf_counter()-start,**meta)
    for rep in range(3):
        for kind,budget in methods[rep:]+methods[:rep]:
            label=f'{case}_{rep}_{kind}_{budget}';save('started_'+label+'.json',dict(case=case,rep=rep,kind=kind,budget=budget,records=count));start=time.perf_counter()
            try:
                bits,counts,row=run(kind,budget,rep);row['success']=True;np.savez_compressed(OUT/('counts_'+label+'.npz'),clicks=bits,**({} if counts is None else dict(counts=counts)))
            except Exception as e:row=dict(success=False,error=str(e),wall_seconds=time.perf_counter()-start)
            row.update(case=case,rep=rep,kind=kind,budget=budget,records=count,inputs=n,outputs=m);rows.append(row);save('row_'+label+'.json',row);print(label,row,flush=True)
        if t is not None:
            aa=np.load(OUT/f'counts_{case}_{rep}_cached_8.npz')['counts'];bb=np.load(OUT/f'counts_{case}_{rep}_uncached_8.npz')['counts'];assert np.array_equal(aa,bb)
    if name=='wide144':
        for kind in ('cached','uncached'):
            start=time.perf_counter();model,meta=make_transport(kind);setup=time.perf_counter()-start
            with model:
                for rep in range(3):
                    r=model.sample(100000,0xb4000000000000+rep*2**40,fixed_total=34);np.savez_compressed(OUT/f'batch_{kind}_{rep}.npz',counts=r['counts']);row=dict(kind=kind,rep=rep,records=100000,fixed_total=34,initial_setup_seconds=setup,sampling_seconds=r['wall_seconds'],**meta);batchrows.append(row);save(f'batch_{kind}_{rep}.json',row);print('batch',row,flush=True)
        for rep in range(3):assert np.array_equal(np.load(OUT/f'batch_cached_{rep}.npz')['counts'],np.load(OUT/f'batch_uncached_{rep}.npz')['counts'])
medians={f'{c}_{k}_{b}':statistics.median(r['wall_seconds'] for r in rows if (r['case'],r['kind'],r['budget'])==(c,k,b)) for c,k,b in set((r['case'],r['kind'],r['budget']) for r in rows) if len([r for r in rows if (r['case'],r['kind'],r['budget'])==(c,k,b) and r['success']])==3}
save('summary.json',dict(completed=True,rows=rows,medians=medians,batches=batchrows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
