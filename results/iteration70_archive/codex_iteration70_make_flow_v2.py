from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PRIVATE=ROOT/'.git'
s=(PRIVATE/'iteration70_flow_v1.cpp').read_text(encoding='utf-8')
workspace='''using EVD70=std::int64_t(*)(int,char,char,std::int64_t,double*,std::int64_t,double*,double*,std::int64_t,std::int64_t*,std::int64_t);
struct EVDWorkspace70 {
    EVD70 function;std::vector<double> work;std::vector<std::int64_t> iwork;
    EVDWorkspace70(int n,EVD70 f):function(f){if(!f)return;Real a=Real::Identity(n,n);Vector values(n);double w=0.;std::int64_t iw=0;auto info=f(102,'V','L',n,a.data(),n,values.data(),&w,-1,&iw,-1);need(info==0&&std::isfinite(w)&&w>=1.&&iw>=1,"LAPACKE workspace query failed");work.resize(std::size_t(std::ceil(w)));iwork.resize(std::size_t(iw));}
};
'''
s=s.replace('struct Evaluation70',workspace+'struct Evaluation70',1)
s=s.replace('int algorithm,double* stats){','int algorithm,double* stats,EVDWorkspace70& workspace){',1)
old='auto start=Clock::now();Eigen::SelfAdjointEigenSolver<Real> solver(x);need(solver.info()==Eigen::Success,"Profiled eigensolver failure");const auto& values=solver.eigenvalues();need(values.minCoeff()>-1.,"Invalid profiled determinant domain");Vector rotated=solver.eigenvectors().transpose()*w;stats[8]+='
new='auto start=Clock::now();Vector values(x.rows());Real basis;if(algorithm==2){need(workspace.function,"Missing LAPACKE function");basis=x;auto info=workspace.function(102,\'V\',\'L\',x.rows(),basis.data(),x.rows(),values.data(),workspace.work.data(),workspace.work.size(),workspace.iwork.data(),workspace.iwork.size());need(info==0,"LAPACKE eigensolver failure");}else{Eigen::SelfAdjointEigenSolver<Real> solver(x);need(solver.info()==Eigen::Success,"Profiled eigensolver failure");values=solver.eigenvalues();basis=solver.eigenvectors();}need(values.minCoeff()>-1.,"Invalid profiled determinant domain");Vector rotated=basis.transpose()*w;stats[8]+='
assert old in s;s=s.replace(old,new)
s=s.replace('algorithm<=1','algorithm<=2').replace('if(algorithm==1)','if(algorithm!=0)').replace('solver.eigenvectors()*damped','basis*damped').replace('*solver.eigenvectors().transpose()','*basis.transpose()').replace('result.w=solver.eigenvectors()*rotated','result.w=basis*rotated')
s=s.replace('API int g70_sample(','API int g70_sample_v2(').replace('API int g70_probe(','API int g70_probe_v2(')
s=s.replace('int selector,int algorithm,const double* mean','int selector,int algorithm,void* eigen_function,const double* mean').replace('double waiting,int algorithm,double* output','double waiting,int algorithm,void* eigen_function,double* output')
s=s.replace('auto& model=*static_cast<Model*>(handle);Vector center','auto& model=*static_cast<Model*>(handle);need(algorithm!=2||eigen_function,"Missing LAPACKE function");EVDWorkspace70 workspace(2*model.n,algorithm==2?reinterpret_cast<EVD70>(eigen_function):nullptr);Vector center')
s=s.replace('int d=2*model.n;Real x','int d=2*model.n;need(algorithm!=2||eigen_function,"Missing LAPACKE function");EVDWorkspace70 workspace(d,algorithm==2?reinterpret_cast<EVD70>(eigen_function):nullptr);Real x')
s=s.replace('algorithm,stats);','algorithm,stats,workspace);')
p=PRIVATE/'iteration70_flow_v2.cpp';assert not p.exists();p.write_bytes(s.encode('utf-8'))
s=(PRIVATE/'codex_iteration70_flow_pilot_v1.py').read_text(encoding='utf-8').replace('iteration70_flow_pilot_v1_20261110','iteration70_flow_pilot_v2_20261110').replace('iteration70_flow_v1.cpp','iteration70_flow_v2.cpp').replace('lib.g70_sample','lib.g70_sample_v2').replace('lib.g70_probe','lib.g70_probe_v2')
s=s.replace('C.c_uint64,C.c_int,C.c_int,DP','C.c_uint64,C.c_int,C.c_int,C.c_void_p,DP').replace('DP,DP,C.c_double,C.c_int,DP','DP,DP,C.c_double,C.c_int,C.c_void_p,DP')
s=s.replace("error=C.create_string_buffer(4096);checks=[];dependencies={}","eigen_function=C.cast(blas.scipy_LAPACKE_dsyevd_work64_,C.c_void_p)\nerror=C.create_string_buffer(4096);checks=[];dependencies={}")
s=s.replace('for algorithm in (0,1):','for algorithm in (0,1,2):').replace("float(d['waiting']),algorithm,ptr(xx)","float(d['waiting']),algorithm,eigen_function,ptr(xx)")
old="a,b=results;row=dict(case=case,step=step,state_error="
new="a=results[0]\n        for algorithm,b in enumerate(results[1:],1):\n            row=dict(case=case,step=step,algorithm=algorithm,state_error="
assert old in s;s=s.replace(old,new)
s=s.replace('((0,1) if repetition==0 else (1,0))','((0,1,2) if repetition==0 else (2,1,0))').replace('selector,algorithm,ptr(mean)','selector,algorithm,eigen_function,ptr(mean)').replace('(algorithm==1 or row[','(algorithm!=0 or row[')
s=s.replace("1:'safeguarded Newton with residual stopping and bisection fallback'", "1:'Newton plus Eigen',2:'Newton plus explicit single-thread ILP64 LAPACKE dsyevd with reusable workspace'")
p=PRIVATE/'codex_iteration70_flow_pilot_v2.py';assert not p.exists();p.write_bytes(s.encode('utf-8'))
