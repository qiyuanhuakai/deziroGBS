// Diagnose replacing the state after a click by its unconditional marginal.
// The input Q = (Gamma + I)/2 uses x coordinates followed by p coordinates.
#include <Eigen/Cholesky>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using Scalar = long double;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Pair = Eigen::Matrix<Scalar, 2, 2>;
using Mean = Eigen::Matrix<Scalar, 2, 1>;

Scalar log_vacuum(const Pair& covariance, const Mean& mean) {
    Eigen::LLT<Pair> factor(covariance);
    if (factor.info() != Eigen::Success) throw std::domain_error("single-port covariance");
    const Pair lower = factor.matrixL();
    const Mean white = lower.triangularView<Eigen::Lower>().solve(mean);
    return -std::log(lower(0,0)) - std::log(lower(1,1)) - white.squaredNorm()/2;
}
}

extern "C" __declspec(dllexport) int reduced95_probe(
    int modes, int first_click, const double* q, const double* mu,
    double min_gamma, double* summary, double* future) {
    if (!q || !mu || !summary || !future || modes < 1 || modes > 2048 ||
        first_click < 0 || first_click >= modes || !(min_gamma > 0) ||
        !std::isfinite(min_gamma)) return 1;
    try {
        const int dim = 2*modes;
        std::vector<int> prefix;
        for (int i = 0; i < first_click; ++i) {
            prefix.push_back(i);
            prefix.push_back(i+modes);
        }
        const int n = static_cast<int>(prefix.size());
        // Whiten prefix-to-port covariances once. Pair queries then cost O(n).
        Matrix whitened = Matrix::Zero(n, dim);
        Vector white_mean(n);
        Scalar log_prefix = 0;
        if (n) {
            Matrix block(n,n), cross(n,dim);
            Vector prefix_mean(n);
            for (int i = 0; i < n; ++i) {
                prefix_mean[i] = mu[prefix[i]];
                for (int j = 0; j < n; ++j) block(i,j) = q[prefix[i]*dim+prefix[j]];
                for (int j = 0; j < dim; ++j) cross(i,j) = q[prefix[i]*dim+j];
            }
            Eigen::LLT<Matrix> factor(block);
            if (factor.info()!=Eigen::Success) return 2;
            const Matrix lower = factor.matrixL();
            whitened = lower.triangularView<Eigen::Lower>().solve(cross);
            white_mean = lower.triangularView<Eigen::Lower>().solve(prefix_mean);
            log_prefix = -lower.diagonal().array().log().sum() - white_mean.squaredNorm()/2;
        }
        auto cov = [&](int row, int column) -> Scalar {
            return q[row*dim+column] - whitened.col(row).dot(whitened.col(column));
        };
        auto mean = [&](int row) -> Scalar {return mu[row] - whitened.col(row).dot(white_mean);};
        auto pair = [&](int a, int b) {
            Pair result;
            result << cov(a,b), cov(a,b+modes), cov(a+modes,b), cov(a+modes,b+modes);
            return result;
        };
        const Pair measured = pair(first_click, first_click);
        const Mean measured_mean(mean(first_click),mean(first_click+modes));
        const Scalar log_p0 = log_vacuum(measured, measured_mean);
        const Scalar p0 = std::exp(log_p0), p1 = -std::expm1(log_p0);
        if (!(p0>0 && p1>0 && p1<=1)) return 3; // impossible click has no posterior
        const Eigen::LLT<Pair> measured_factor(measured);
        Scalar max_witness = 0, energy = 0;
        int best_port = -1;
        for (int j = first_click+1; j < modes; ++j) {
            const Pair marginal = pair(j,j), cross = pair(j,first_click);
            const Mean marginal_mean(mean(j),mean(j+modes));
            const Pair white = measured_factor.matrixL().solve(cross.transpose());
            const Pair shift_cov = white.transpose()*white;
            const Mean shift = cross * measured_factor.solve(measured_mean);
            const Scalar marginal_log = log_vacuum(marginal,marginal_mean);
            const Scalar zero_log = log_vacuum(marginal-shift_cov,marginal_mean-shift);
            const Scalar marginal_vac = std::exp(marginal_log);
            const Scalar delta = -(p0/p1)*marginal_vac*std::expm1(zero_log-marginal_log);
            const Scalar posterior = marginal_vac+delta;
            if (!std::isfinite(posterior) || posterior < -1e-11L || posterior > 1+1e-11L) return 4;
            energy += shift_cov.trace() + shift.squaredNorm();
            if (std::abs(delta)>max_witness) {max_witness=std::abs(delta); best_port=j;}
            future[3*j]=double(marginal_vac);
            future[3*j+1]=double(posterior);
            future[3*j+2]=double(delta);
        }
        const Scalar upper = std::min(p0,(p0/p1)*std::sqrt(-std::expm1(-energy/(2*min_gamma))));
        if (!std::isfinite(upper) || max_witness>upper+1e-10L) return 5;
        summary[0]=double(std::exp(log_prefix)); summary[1]=double(p1);
        summary[2]=double(max_witness); summary[3]=double(upper);
        summary[4]=double(energy); summary[5]=best_port;
        return 0;
    } catch (const std::exception&) {return 6;}
}
