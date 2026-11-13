// Re-express a compressed Gaussian as a random displaced pure Gaussian.
// For V=I+2X, Vp=sqrt(V)|sqrt(V) Omega sqrt(V)|^{-1}sqrt(V).
// C=(V-Vp)/2 is the displacement covariance. Negative eigenvalues are not
// clipped: unresolved decompositions fall back to the prior Gaussian flow.
struct Split75 {Real rotation,noise_factor;Vector t;double residual=0;bool valid=false;};
static Split75 decompose75(const Real& x){
    int d=x.rows(),n=d/2;Split75 out;Real v=Real::Identity(d,d)+2*x,omega=Real::Zero(d,d);omega.topRightCorner(n,n).setIdentity();omega.bottomLeftCorner(n,n)=-Real::Identity(n,n);
    Eigen::SelfAdjointEigenSolver<Real> ve(v);need(ve.info()==Eigen::Success&&ve.eigenvalues().minCoeff()>0,"Resplit covariance eigensolver");
    Real root=ve.eigenvectors()*ve.eigenvalues().cwiseSqrt().asDiagonal()*ve.eigenvectors().transpose(),a=root*omega*root,k=-a*a;k=(k+k.transpose()).eval()/2;
    Eigen::SelfAdjointEigenSolver<Real> ke(k);need(ke.info()==Eigen::Success&&ke.eigenvalues().minCoeff()>0,"Resplit symplectic square eigensolver");
    if(ke.eigenvalues().minCoeff()<1)return out;
    Real vp=root*ke.eigenvectors()*ke.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal()*ke.eigenvectors().transpose()*root;vp=(vp+vp.transpose()).eval()/2;
    Eigen::SelfAdjointEigenSolver<Real> pe(vp);need(pe.info()==Eigen::Success&&pe.eigenvalues().minCoeff()>0,"Resplit pure eigensolver");
    Real rotation=Real::Zero(d,d);int accepted=0;
    for(int col=0;col<d&&accepted<n;++col){Vector q=pe.eigenvectors().col(col);
        for(int pass=0;pass<2;++pass)for(int j=0;j<accepted;++j){q-=rotation.col(j)*rotation.col(j).dot(q);q-=rotation.col(n+j)*rotation.col(n+j).dot(q);}
        double norm=q.norm();if(norm<1e-7)continue;q/=norm;rotation.col(accepted)=q;rotation.col(n+accepted)=-omega*q;++accepted;
    }
    if(accepted!=n)return out;Vector t(n),diagonal(d);
    for(int j=0;j<n;++j){double lambda=rotation.col(j).dot(vp*rotation.col(j));t[j]=(1-lambda)/(1+lambda);if(!(std::isfinite(t[j])&&std::abs(t[j])<1))return out;diagonal[j]=-t[j]/(1+t[j]);diagonal[n+j]=t[j]/(1-t[j]);}
    Real xp=rotation*diagonal.asDiagonal()*rotation.transpose(),reconstructed=Real::Identity(d,d)+2*xp;
    double residual=(reconstructed-vp).cwiseAbs().maxCoeff()/std::max(1.,vp.cwiseAbs().maxCoeff());
    if(residual>1e-10||(rotation.transpose()*rotation-Real::Identity(d,d)).cwiseAbs().maxCoeff()>1e-10)return out;
    Real noise=x-xp;noise=(noise+noise.transpose()).eval()/2;Eigen::SelfAdjointEigenSolver<Real> ne(noise);need(ne.info()==Eigen::Success,"Resplit noise eigensolver");if(ne.eigenvalues().minCoeff()<0)return out;
    out.rotation=rotation;out.t=t;out.noise_factor=ne.eigenvectors()*ne.eigenvalues().cwiseSqrt().asDiagonal();out.residual=residual;out.valid=true;return out;
}
struct ResplitStats75 {double attempts=0,successes=0,residual=0;};
static bool resplit75(ClickModel70& c,State73& state,Vector& t,std::mt19937_64& rng,ResplitStats75& stats){
    ++stats.attempts;need(state.factors.empty(),"Resplit requires compressed Gaussian");auto split=decompose75(state.x);if(!split.valid)return false;
    int n=c.model->n,m=c.model->m;std::normal_distribution<double> normal;Vector z(2*n);for(int j=0;j<2*n;++j)z[j]=normal(rng);Vector nextmean=split.rotation.transpose()*(state.mean+split.noise_factor*z);
    Real nexta=split.rotation.transpose()*c.a,nextb=split.rotation.transpose()*c.b;Complex nextv(m,n);
    for(int i=0;i<m;++i)for(int j=0;j<n;++j)nextv(i,j)={nexta(j,i),nextb(j,i)};
    need((nextv.adjoint()*nextv-Complex::Identity(n,n)).norm()<1e-8,"Resplit passive isometry residual");
    c.model->v=nextv;for(int i=0;i<m;++i)for(int j=0;j<n;++j){auto v=nextv(i,j);c.a(j,i)=v.real();c.a(j+n,i)=-v.imag();c.b(j,i)=v.imag();c.b(j+n,i)=v.real();}
    t=split.t;state.mean=nextmean;state.x.setZero();for(int j=0;j<n;++j){state.x(j,j)=-t[j]/(1+t[j]);state.x(n+j,n+j)=t[j]/(1-t[j]);}
    ++stats.successes;stats.residual=std::max(stats.residual,split.residual);return true;
}
