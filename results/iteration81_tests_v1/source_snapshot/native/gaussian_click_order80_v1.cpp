// Input-only detector order heuristics. Output labels are restored by caller.
// Zero-mean covariance convention: vacuum I, Q=(cov+I)/2.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>
#define API extern "C" __declspec(dllexport)
API int order80_build(int m,const double* covariance,int* orders,double* vacuum,double* correlations,char* message,int cap){try{
 auto need=[](bool good,const char* why){if(!good)throw std::runtime_error(why);};need(m>=2&&m<=512&&covariance&&orders&&vacuum&&correlations,"Invalid ordering arguments");
 using Matrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
 Eigen::Map<const Matrix> input(covariance,2*m,2*m);need(input.allFinite()&&(input-input.transpose()).cwiseAbs().maxCoeff()<1e-10,"Ordering covariance must be finite and symmetric");
 Eigen::MatrixXd Q=(input+Eigen::MatrixXd::Identity(2*m,2*m))*.5,W=Eigen::MatrixXd::Zero(m,m);
 auto logvac=[&](const std::vector<int>& modes){int size=2*modes.size();Eigen::Matrix<long double,Eigen::Dynamic,Eigen::Dynamic> block(size,size);std::vector<int> ids=modes;for(int k:modes)ids.push_back(k+m);for(int i=0;i<size;++i)for(int j=0;j<size;++j)block(i,j)=Q(ids[i],ids[j]);Eigen::LLT<decltype(block)> factor(block);need(factor.info()==Eigen::Success,"Ordering Q principal block is not positive definite");auto L=factor.matrixL().toDenseMatrix();long double value=0;for(int i=0;i<size;++i)value-=std::log(L(i,i));need(std::isfinite(value)&&value<=1e-13L,"Invalid ordering vacuum probability");return value;};
 for(int i=0;i<m;++i)vacuum[i]=double(std::exp(logvac({i})));
 for(int i=0;i<m;++i)for(int j=0;j<i;++j){long double joint=std::exp(logvac({i,j}));double correlation=double(joint-(long double)vacuum[i]*vacuum[j]);correlations[i*m+j]=correlations[j*m+i]=correlation;W(i,j)=W(j,i)=std::abs(correlation);}
 for(int i=0;i<m;++i)correlations[i*m+i]=vacuum[i]*(1-vacuum[i]);
 std::vector<int> ids(m);std::iota(ids.begin(),ids.end(),0);auto store=[&](int method){std::copy(ids.begin(),ids.end(),orders+method*m);};store(0);
 std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return vacuum[a]>vacuum[b];});store(1);
 std::iota(ids.begin(),ids.end(),0);std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return vacuum[a]<vacuum[b];});store(2);
 Eigen::VectorXd degree=W.rowwise().sum();Eigen::MatrixXd laplacian=degree.asDiagonal().toDenseMatrix()-W;Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(laplacian);need(eig.info()==Eigen::Success,"Ordering spectral solve failed");Eigen::VectorXd fiedler=eig.eigenvectors().col(1);for(int i=0;i<m;++i)if(std::abs(fiedler[i])>1e-12){if(fiedler[i]<0)fiedler=-fiedler;break;}
 std::iota(ids.begin(),ids.end(),0);std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return fiedler[a]<fiedler[b];});store(3);
 std::vector<bool> used(m,false);int current=std::max_element(vacuum,vacuum+m)-vacuum;ids.clear();
 for(int k=0;k<m;++k){ids.push_back(current);used[current]=true;int next=-1;for(int j=0;j<m;++j)if(!used[j]&&(next<0||W(current,j)>W(current,next)))next=j;current=next;}store(4);
 ids.resize(m);std::iota(ids.begin(),ids.end(),0);std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return degree[a]<degree[b];});store(5);return 0;
}catch(const std::exception& e){if(message&&cap>0){auto n=std::min(std::strlen(e.what()),std::size_t(cap-1));std::memcpy(message,e.what(),n);message[n]=0;}return 1;}}
