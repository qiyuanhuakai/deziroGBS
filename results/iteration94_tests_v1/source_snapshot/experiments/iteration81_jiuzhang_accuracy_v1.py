"""Predeclared eight-bin marginals from the first high-power Jiuzhang3 file.
Two common independent spectral sectors are the explicit reference model inferred
from the data README. Also quantify main-only and source-distinct-minor sensitivity;
do not silently identify a model with an exact description of experimental noise.
Frugal and direct samplers each sample both Gaussian sectors then OR their bits.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,ctypes as C,subprocess,argparse,time,math,zipfile,xml.etree.ElementTree as ET
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.frugal_law import full_law
from gbslab.native_click_branches import NativeBranchClickSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
base=ROOT/'results/iteration81_jiuzhang_extract_v1';external=ROOT.parent/'external_sources/jiuzhang3_iteration81_sample_v1'
mtx=base/'extracted_0/jiuzhang3/matrix/high power.mtx';xlsx=base/'extracted_2/jiuzhang3/sqz parameter/high power.xlsx';raw=external/'extracted_3/jiuzhang3/data/high power/high power (1).bin'
ports=np.array([list(range(0,64,8)),list(range(0,1152,144)),list(range(7,1152,144))],np.int32);pilot=100000
gcc=Path(shutil.which('g++'));runtime=[gcc.parent,Path(shutil.which('nvcc')).parent];handles=[os.add_dll_directory(str(d)) for d in runtime]
bins={k:ROOT/v for k,v in dict(frugal='results/iteration81_prefix_build_v1/prefix.dll',frugal_sample='results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll',branch='results/iteration71_branch_pilot_v1/branches.dll',cpu='results/iteration72_cuda_pilot_v1/cpu.dll',gpu='results/iteration74_fused_pilot_v2/philox.dll').items()}
source=ROOT/'native/jiuzhang_histogram81_v1.cpp';save('plan.json',dict(scope=__doc__,ports=ports.tolist(),pilot=pilot,reference_models=['two_common_sectors','main_only','main_and_25_distinct_minor'],frugal_cutoffs=[1,2,3,4,5],direct_budgets=[1,2,4,8,16,32,128],timing_count=8192,timing_repetitions=3,threads=16,hashes={str(p):H(p) for p in [Path(__file__),source,mtx,xlsx,raw,*bins.values()]},iid_confidence=.99,confidence_bounds=18))
assert H(raw)=='c786e6e64f76a599479eda75c7fdedf6906a3b41040eec49e5ea89caeb714e74'
command=[str(gcc),'-O3','-std=c++20','-shared','-fopenmp',str(source),'-o',str(OUT/'hist.dll')];r=subprocess.run(command,capture_output=True,text=True);save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0
lib=C.CDLL(str(OUT/'hist.dll'));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint64);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP)
fn=lib.jz81_hist;fn.argtypes=[BP,C.c_uint64,C.c_int,IP,C.c_uint64,C.c_int,UP,UP];fn.restype=C.c_int
assert raw.stat().st_size%144==0;N=raw.stat().st_size//144;data=np.memmap(raw,dtype=np.uint8,mode='r',shape=(N,144));hist=np.zeros((3,2,256),np.uint64);total=np.zeros(1153,np.uint64)
start=time.perf_counter();assert fn(data.ctypes.data_as(BP),N,3,ports.ctypes.data_as(IP),pilot,16,hist.ctypes.data_as(UP),total.ctypes.data_as(UP))==0;histtime=time.perf_counter()-start
assert np.all(hist[:,0].sum(1)==pilot) and np.all(hist[:,1].sum(1)==N-pilot) and total.sum()==N
# Independent parser checks all selected bits on the first 4096 rows.
bits=np.unpackbits(np.asarray(data[:4096]),axis=1)
for g in range(3):
 check=np.bincount((bits[:,ports[g]]*(1<<np.arange(8))).sum(1),minlength=256);native=np.zeros((3,2,256),np.uint64);nt=np.zeros(1153,np.uint64)
 assert fn(data.ctypes.data_as(BP),4096,3,ports.ctypes.data_as(IP),1,1,native.ctypes.data_as(UP),nt.ctypes.data_as(UP))==0
 assert np.array_equal(check,native[g].sum(0))
np.savez_compressed(OUT/'histograms.npz',ports=ports,histograms=hist,total_click_histogram=total)
save('data_summary.json',dict(records=N,pilot=pilot,holdout=N-pilot,histogram_seconds=histtime,mean_total_clicks=float(np.dot(total,np.arange(1153))/N),maximum_total_clicks=int(np.flatnonzero(total)[-1]),parser_crosscheck_rows=4096))
rawmatrix=np.loadtxt(mtx,skiprows=3);T=(rawmatrix[:,0]+1j*rawmatrix[:,1]).reshape(1152,50,order='F')
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(xlsx) as z:sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
rr=np.zeros((25,2))
for cell in sheet.findall('.//s:c',ns):
 value=cell.find('s:v',ns)
 if value is not None:
  ref=cell.attrib['r'];rr[int(ref[1:])-1,ord(ref[0])-ord('A')]=float(value.text)
assert np.all(rr>0) and np.all(rr[:,0]>rr[:,1]);patterns=np.array([[(s>>j)&1 for j in range(8)] for s in range(256)],np.uint8)
def covariance(t,r):
 n=t.shape[0];nn=(t*np.repeat(np.sinh(r)**2,2))@t.conj().T;mm=(t*np.repeat(np.sinh(r)*np.cosh(r),2))@t[:,np.arange(50)^1].T
 c=np.block([[np.eye(n)+2*(nn+mm).real,2*(mm-nn).imag],[2*(mm+nn).imag,np.eye(n)+2*(nn-mm).real]])
 assert np.max(abs(c-c.T))<1e-12
 omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
 assert np.linalg.eigvalsh(c+1j*omega).min()>-1e-10
 return np.ascontiguousarray(c)
def vacuum(c):
 v=np.ones(256,np.longdouble)
 for mask in range(1,256):
  idx=[j for j in range(8) if mask>>j&1];idx+= [8+j for j in idx];sg,ld=np.linalg.slogdet((c[np.ix_(idx,idx)]+np.eye(len(idx)))/2);assert sg>0;v[mask]=np.exp(-np.longdouble(ld)/2)
 return v
def invert(v):
 law=v[np.arange(256)^255].copy()
 for bit in range(8):
  for mask in range(256):
   if mask>>bit&1:law[mask]-=law[mask^(1<<bit)]
 assert law.min()>-1e-13 and abs(law.sum()-1)<1e-12
 return law
def combine(laws):
 # OR convolution using no-click moments; never sum the covariances.
 vv=np.ones(256,np.longdouble)
 for law in laws:vv*=np.array([law[(np.arange(256)&mask)==0].sum() for mask in range(256)])
 return invert(vv)
flib=C.CDLL(str(bins['frugal']));treefn=flib.fc81_tree;treefn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];treefn.restype=C.c_int
slib=C.CDLL(str(bins['frugal_sample']));sfn=slib.fc73_sample;sfn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,DP,BP,DP,C.c_char_p,C.c_int];sfn.restype=C.c_int
rows=[];inputs=[]
for g,ids in enumerate(ports):
 covs=[covariance(T[ids],rr[:,s]) for s in range(2)];vv=[vacuum(c) for c in covs];targets={'two_common_sectors':invert(vv[0]*vv[1]),'main_only':invert(vv[0])}
 distinct=vv[0].copy()
 for s in range(25):
  r=np.zeros(25);r[s]=rr[s,1];distinct*=vacuum(covariance(T[ids],r))
 targets['main_and_25_distinct_minor']=invert(distinct);target=targets['two_common_sectors'];inputs.append(covs)
 np.savez_compressed(OUT/f'input_{g}.npz',covariances=np.array(covs),T=T[ids],squeezing=rr,ports=ids,patterns=patterns,**targets)
 for name,t in targets.items():
  empirical=hist[g,1].astype(np.longdouble)/(N-pilot);pilotlaw=hist[g,0].astype(np.longdouble)/pilot;event=pilotlaw>t;diff=float(empirical[event].sum()-t[event].sum());delta=.01/18
  lower=max(0,diff-math.sqrt(math.log(1/delta)/(2*(N-pilot))));radius=math.sqrt((256*math.log(2)+math.log(1/delta))/(2*(N-pilot)));tv=float(abs(empirical-t).sum()/2)
  row=dict(group=g,method='experiment',reference=name,empirical_tv=tv,nominal_joint99_lower=lower,nominal_joint99_upper=min(1,tv+radius),reference_difference_tv=float(abs(t-target).sum()/2),holdout=N-pilot,witness_event=np.flatnonzero(event).tolist());rows.append(row);save(f'experiment_{g}_{name}.json',row)
 for method,budgets in [('frugal',[1,2,3,4,5]),('direct',[1,2,4,8,16,32,128])]:
  for budget in budgets:
   laws=[];gpu=[]
   for s,c in enumerate(covs):
    if method=='frugal':
     q=np.zeros(255);prev=q.copy();proposed=q.copy();stats=np.zeros(3);err=C.create_string_buffer(4096);assert treefn(8,budget,8,16,ptr(c),ptr(q),ptr(prev),ptr(proposed),ptr(stats),err,len(err))==0,err.value
     tree=[dict(prefix=''.join(str((bits>>j)&1) for j in range(depth)),threshold_hex=float(q[(1<<depth)-1+bits]).hex()) for depth in range(8) for bits in range(1<<depth)];dist=full_law(8,tree,model='uint64');law=np.array([float(v) for v in dist['distribution']],np.longdouble);save(f'tree_{g}_{budget}_{s}.json',dict(tree=tree,nonfinite=list(dist['reachable_nonfinite_prefixes']),saturated=list(dist['reachable_saturated_prefixes'])));np.savez_compressed(OUT/f'frugal_{g}_{budget}_{s}.npz',law=law,thresholds=q,previous=prev,proposed=proposed,stats=stats)
    else:
     with NativeBranchClickSampler(np.eye(8), (c-np.eye(16))/2,bins['branch'],branch_budget=budget,runtime_dirs=runtime) as model:law=np.array([model.prefix_probability(b)['probability'] for b in patterns],np.longdouble)
     if budget in (4,8):
      with NativePhiloxClickSampler(np.eye(8),(c-np.eye(16))/2,bins['cpu'],gpu_library=bins['gpu'],branch_budget=budget,runtime_dirs=runtime) as model:replay=model.replay(np.zeros((256,16)),np.where(patterns,0.,np.nextafter(1.,0.)))
      assert np.array_equal(replay['clicks'],patterns);prob=replay['conditional_click_probabilities'];glaw=np.prod(np.where(patterns,prob,1-prob),axis=1);assert np.max(abs(glaw-law))<1e-9;gpu.append(glaw)
     np.savez_compressed(OUT/f'direct_{g}_{budget}_{s}.npz',law=law)
    laws.append(law)
   law=combine(laws);row=dict(group=g,method=method+'_two_sectors_or',budget=budget,numerical_tv=float(abs(law-target).sum()/2),empirical_tv=float(abs(law-hist[g,1]/(N-pilot)).sum()/2));rows.append(row);np.savez_compressed(OUT/f'law_{g}_{method}_{budget}.npz',law=law);save(f'row_{g}_{method}_{budget}.json',row)
   if gpu:glaw=combine(gpu);assert np.max(abs(glaw-law))<1e-9;np.savez_compressed(OUT/f'law_{g}_gpu_{budget}.npz',law=glaw)
 print('Accuracy group completed',g,[(r['method'],r.get('budget'),r.get('numerical_tv',r.get('empirical_tv'))) for r in rows if r['group']==g],flush=True)
save('accuracy_summary.json',dict(completed=True,rows=rows))
# No other heavy process is launched during this serial timing phase. These are
# eight-bin subsystem samplers, NOT full 1152-bin hardware-task comparisons.
timings=[];count=8192;panels=[('frugal',2),('frugal',3),('frugal',5),('direct',4),('direct',8)]
for rep in range(3):
 for method,budget in panels[rep:]+panels[:rep]:
  start=time.perf_counter();outputs=[]
  for s,c in enumerate(inputs[0]):
   seed=810100+rep*10+s
   if method=='frugal':
    out=np.zeros((count,8),np.uint8);stats=np.zeros(7);err=C.create_string_buffer(4096);assert sfn(8,budget,count,seed,16,ptr(c),out.ctypes.data_as(BP),ptr(stats),err,len(err))==0,err.value
   else:
    with NativePhiloxClickSampler(np.eye(8),(c-np.eye(16))/2,bins['cpu'],gpu_library=bins['gpu'],branch_budget=budget,runtime_dirs=runtime) as model:out=model.sample(count,seed)['clicks']
   outputs.append(out)
  result=np.bitwise_or(*outputs);elapsed=time.perf_counter()-start;assert result.shape==(count,8);row=dict(rep=rep,method=method,budget=budget,records=count,seconds=elapsed);timings.append(row);save(f'time_{rep}_{method}_{budget}.json',row);np.savez_compressed(OUT/f'samples_{rep}_{method}_{budget}.npz',clicks=result);print(row,flush=True)
save('summary.json',dict(completed=True,accuracy=rows,timings=timings,scope='Three fixed eight-detector marginals; speed samples only group zero. CPU Frugal samples two Gaussian sectors independently then OR; direct GPU does likewise. Not full hardware throughput.',records=N))
save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
