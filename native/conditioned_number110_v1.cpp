#include <Eigen/Dense>
#include <algorithm>
#include <bit>
#include <cmath>
#include <vector>

using Matrix110=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
extern "C" int conditioned110(int bits,int d,int bins,int workers,const double* eigen,const double* head,double* result){
    if(bits<1||bits>12||d<1||bins<1||workers<1)return 1;
    Eigen::Map<const Eigen::VectorXd> e(eigen,d);Eigen::Map<const Matrix110> h(head,2*bits,d);
    int errors=0;
    #pragma omp parallel for num_threads(workers) schedule(dynamic) reduction(+:errors)
    for(int mask=0;mask<(1<<bits);++mask){
        Eigen::VectorXd ev=e;long double logvac=0;
        if(mask){
            int count=std::popcount(unsigned(mask));Eigen::MatrixXd v(2*count,d);
            for(int k=0,j=0;k<bits;++k)if(mask&(1<<k)){v.row(j)=h.row(k);v.row(j+count)=h.row(k+bits);++j;}
            Eigen::MatrixXd we=v*e.asDiagonal();
            Eigen::MatrixXd identity=Eigen::MatrixXd::Identity(2*count,2*count);
            Eigen::LLT<Eigen::MatrixXd> q(identity+we*v.transpose());
            if(q.info()!=Eigen::Success){++errors;continue;}
            Eigen::VectorXd diag=q.matrixL().toDenseMatrix().diagonal();logvac=-diag.array().log().sum();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ortho(identity-v*v.transpose());
            if(ortho.info()!=Eigen::Success||ortho.eigenvalues().minCoeff() < -1.e-10){++errors;continue;}
            auto vec=ortho.eigenvectors();Eigen::VectorXd inv=(1.+ortho.eigenvalues().cwiseMax(0.).array().sqrt()).inverse();
            Eigen::MatrixXd root=Eigen::MatrixXd::Identity(d,d)-v.transpose()*vec*inv.asDiagonal()*vec.transpose()*v;
            Eigen::MatrixXd y=e.asDiagonal();y-=we.transpose()*q.solve(we);
            Eigen::MatrixXd projected=root*y*root;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(projected,Eigen::EigenvaluesOnly);
            if(solver.info()!=Eigen::Success){++errors;continue;}ev=solver.eigenvalues();
        }
        std::vector<long double> k(d),hprev(d);
        for(int i=0;i<d;++i){long double value=ev[i];logvac-=std::log1p(value)/2;k[i]=value/(1+value);}
        long double p=std::exp(logvac);auto* out=result+size_t(mask)*bins;out[0]=double(p);
        for(int n=1;n<bins;++n){
            long double sum=0;
            for(int i=0;i<d;++i){hprev[i]=p+k[i]*hprev[i];sum+=k[i]*hprev[i];}
            p=sum/(2*n);out[n]=double(p);
        }
    }
    return errors;
}
