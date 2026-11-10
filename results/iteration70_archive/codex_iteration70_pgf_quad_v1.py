from pathlib import Path
import os,sys,ctypes as C,hashlib,json,shutil,subprocess,time,math,itertools,functools
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import mpmath as mp
OUT=ROOT/'.git/iteration70_pgf_quad_v1_20261110';OUT.mkdir(exist_ok=False)
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
source=ROOT/'.git/iteration70_pgf_quad_v1.cpp';gcc=Path(shutil.which('g++'));sources=[Path(__file__),source]
for p in sources:shutil.copy2(p,OUT/p.name)
command=[str(gcc),'-O3','-march=native','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math',str(source),'-lquadmath','-o',str(OUT/'oracle.dll')]
cases=[(.7,.1,16),(.7,.3,24),(.7,.65,32),(1.1,.65,48),(1.4,.43,48)]
save('plan.json',dict(source_hashes={p.name:H(p) for p in sources},compiler_sha256=H(gcc),command=command,cases=cases,new_input_seed=701001,scope='Complete joint four-mode PNR simplex plus one untruncated tail bin. Native IEEE binary128 determinant-coefficient recurrence, 100-digit determinant inputs and independent low-order Isserlis/higher-order coefficient checks. Oracle size grows combinatorially in modes, not a scalable arbitrary-mode sampler. Numerical comparisons are not interval-arithmetic certificates. Old four physical inputs are frozen iteration69 fixtures; the fifth uses a fresh isometry and Jiuzhang3 representative r/eta.'))
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));print('build',build.returncode,flush=True);assert build.returncode==0
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'oracle.dll'));U32=C.POINTER(C.c_uint32);U8=C.POINTER(C.c_ubyte);DP=C.POINTER(C.c_double)
lib.g70_pgf.argtypes=[C.c_int,C.c_int,C.c_char_p,C.c_uint64,U32,DP,U8,C.c_char_p,C.c_int,C.c_char_p,C.c_int]
mp.mp.dps=100;rows=[];dependencies={}
def decode128(raw):
    bits=int.from_bytes(bytes(raw),'little');sign=-1 if bits>>127 else 1;exp=(bits>>112)&0x7fff;mantissa=bits&((1<<112)-1)
    assert exp!=0x7fff
    if exp:mantissa+=1<<112;power=exp-16383-112
    else:power=1-16383-112
    return sign*mp.mpf(mantissa)*mp.power(2,power)
for case,(r,eta,cutoff) in enumerate(cases):
    if case<4:
        path=ROOT/'results/iteration69_archive/iteration69_quality_pilot_20261110'/f'input_{case}.npz';dependencies[path.relative_to(ROOT).as_posix()]=H(path);data=np.load(path);v=data['v'];cov=data['covariance']
    else:
        rng=np.random.default_rng(701001);q,rr=np.linalg.qr(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)));v=np.ascontiguousarray(q*(np.diag(rr)/abs(np.diag(rr)))[None,:]);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);number=np.sinh(r)**2;pair=np.sinh(r)*np.cosh(r);x=np.diag([eta*(number+pair)]*2+[eta*(number-pair)]*2);cov=np.eye(8)+2*o@x@o.T;cov=(cov+cov.T)/2
    k=mp.matrix([[(mp.mpf(float(cov[i,j]))-int(i==j))/2 for j in range(8)] for i in range(8)]);a=mp.eye(8)+k;constant=mp.det(a);b=k*a**-1;coeff=[mp.mpf(0)]*81
    for mask in range(256):
        indices=[i for i in range(8) if mask>>i&1];powers=[int(bool(mask>>j&1))+int(bool(mask>>(j+4)&1)) for j in range(4)];label=sum(p*3**j for j,p in enumerate(powers));minor=mp.det(mp.matrix([[b[i,j] for j in indices] for i in indices])) if indices else mp.mpf(1);coeff[label]+=constant*((-1)**len(indices))*minor
    text='\n'.join(mp.nstr(c,100) for c in coeff);(OUT/f'coefficients_{case}.txt').write_bytes(text.encode('ascii'));size=math.comb(cutoff+4,4);codes=np.zeros(size,np.uint32);prob=np.zeros(size);quad=np.zeros((size,16),np.uint8);summary=C.create_string_buffer(16384);error=C.create_string_buffer(4096)
    start=time.perf_counter();rc=lib.g70_pgf(4,cutoff,text.encode(),size,codes.ctypes.data_as(U32),prob.ctypes.data_as(DP),quad.ctypes.data_as(U8),summary,len(summary),error,len(error));seconds=time.perf_counter()-start;save(f'native_{case}.json',dict(returncode=rc,error=error.value.decode(),seconds=seconds,summary=json.loads(summary.value) if rc==0 else None));assert rc==0,error.value;info=json.loads(summary.value);np.savez_compressed(OUT/f'target_{case}.npz',codes=codes,probabilities=prob,binary128=quad,covariance=cov,v=v,cutoff=cutoff,r=r,eta=eta)
    ids={int(code):i for i,code in enumerate(codes)};base=cutoff+1
    def code(counts):return sum(n*base**j for j,n in enumerate(counts))
    # Independent Isserlis expansion of det(I-ZB)^(-1/2), through four photons.
    @functools.lru_cache(None)
    def wick(indices):
        if not indices:return mp.mpf(1)
        return sum(b[indices[0],indices[j]]*wick(indices[1:j]+indices[j+1:]) for j in range(1,len(indices)))
    def isserlis(counts):
        value=mp.mpf(0)
        for selected in itertools.product(*(range(n+1) for n in counts)):
            indices=tuple(sorted([j for j in range(4) for _ in range(2*selected[j])]+[j+4 for j in range(4) for _ in range(2*(counts[j]-selected[j]))]));factor=math.prod(math.comb(n,t) for n,t in zip(counts,selected));value+=factor*wick(indices)
        return value/(mp.sqrt(constant)*mp.power(2,sum(counts))*math.prod(math.factorial(n) for n in counts))
    terms=[]
    for label,c in enumerate(coeff[1:],1):
        digits=tuple((label//3**j)%3 for j in range(4));terms.append((digits,sum(digits),c))
    @functools.lru_cache(None)
    def reference(counts):
        degree=sum(counts)
        if degree==0:return 1/mp.sqrt(constant)
        value=mp.mpf(0)
        for digits,total,c in terms:
            if total<=degree and all(l<=n for l,n in zip(digits,counts)):value+=(degree-mp.mpf(total)/2)*c*reference(tuple(n-l for n,l in zip(counts,digits)))
        return -value/(degree*constant)
    errors=[];independent_checks=0
    for counts in itertools.product(range(5),repeat=4):
        if sum(counts)>4:continue
        actual=decode128(quad[ids[code(counts)]]);expected=isserlis(counts);errors.append(abs(actual-expected));independent_checks+=1
    selected=[tuple(cutoff if j==mode else 0 for j in range(4)) for mode in range(4)]+[(cutoff//4,)*4,(cutoff//2,cutoff//2,0,0)]
    for counts in selected:errors.append(abs(decode128(quad[ids[code(counts)]])-reference(counts)))
    max_error=max(errors);assert max_error<mp.mpf('1e-28'),max_error
    # Exact equal-input total-count generating function, including its tail.
    t=mp.tanh(mp.mpf(r));e=mp.mpf(eta);q=t*t;z=1-e;bp=t*e/(1-t*z);bm=-t*e/(1+t*z);cp=(1-q)/(2*(1-t*z));cm=(1-q)/(2*(1+t*z));expected_total=[cp*bp**j+cm*bm**j for j in range(cutoff+1)];tail=cp*bp**(cutoff+1)/(1-bp)+cm*bm**(cutoff+1)/(1-bm)
    total_errors=[abs(mp.mpf(observed)-expected) for observed,expected in zip(info['total_count_probabilities'],expected_total)];assert max(total_errors)<mp.mpf('2e-13'),max(total_errors)
    row=dict(case=case,r=r,eta=eta,cutoff=cutoff,states=size,seconds=seconds,independent_isserlis_patterns=independent_checks,high_order_reference_patterns=len(selected),maximum_binary128_reference_error=str(max_error),maximum_total_count_error=str(max(total_errors)),analytic_target_tail=str(tail),native_mass=info['mass'],passed=True);rows.append(row);save(f'validation_{case}.json',row);print(row,flush=True)
save('summary.json',dict(passed=all(r['passed'] for r in rows),rows=rows,dependencies=dependencies,oracle_sha256=H(OUT/'oracle.dll')));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
