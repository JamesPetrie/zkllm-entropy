#include "zknn/zknormalcdf.cuh"
#include <stdexcept>

// ── Table initialisation ──────────────────────────────────────────────────────

// mvals[tid] = round(Phi(tid / sigma_eff) * scale_out)
// Phi(x) = erfc(-x / sqrt(2)) / 2   (standard-normal CDF)
// tid in [0, len): represents integer diff value d = tid.
KERNEL void zknormalcdf_init_mvals_kernel(Fr_t* mvals, double sigma_eff, uint scale_out, uint len) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        double d    = (double)tid;
        double x    = d / sigma_eff;
        double cdf  = 0.5 * erfc(-x / sqrt(2.0));
        long scaled = (long)(cdf * (double)scale_out + 0.5);
        mvals[tid]  = long_to_scalar(scaled);
    }
}

static FrTensor make_cdf_mvals(uint precision, uint scale_out, double sigma_eff) {
    uint len = 1u << precision;
    FrTensor mvals(len);
    zknormalcdf_init_mvals_kernel<<<(len + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        mvals.gpu_data, sigma_eff, scale_out, len);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("zknormalcdf_init_mvals_kernel: ") + cudaGetErrorString(err));
    return mvals;
}

// ── Class implementation ──────────────────────────────────────────────────────

zkNormalCDF::zkNormalCDF(uint precision, uint scale_out, double sigma_eff)
    : precision(precision), scale_out(scale_out), sigma_eff(sigma_eff),
      lookup(/*low=*/0, /*len=*/1u << precision, make_cdf_mvals(precision, scale_out, sigma_eff))
{}

pair<FrTensor, FrTensor> zkNormalCDF::compute(const FrTensor& diffs) {
    return lookup(diffs);
}

Fr_t zkNormalCDF::prove(const FrTensor& diffs, const FrTensor& cdf_values, const FrTensor& m,
                        const Fr_t& r, const Fr_t& alpha, const Fr_t& beta,
                        const vector<Fr_t>& u, const vector<Fr_t>& v,
                        vector<Polynomial>& proof) {
    return lookup.prove(diffs, cdf_values, m, r, alpha, beta, u, v, proof);
}
