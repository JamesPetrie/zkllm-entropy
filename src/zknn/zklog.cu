#include "zknn/zklog.cuh"
#include <stdexcept>

// ── Table initialisation ──────────────────────────────────────────────────────

// mvals[tid] = round(-log2((tid+1) / 2^precision) * scale_out)
//            = round((precision - log2(tid+1)) * scale_out)
// tid in [0, len), entry represents input index p_idx = tid + 1.
KERNEL void zklog_init_mvals_kernel(Fr_t* mvals, uint precision, uint scale_out, uint len) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        double p_idx = (double)(tid + 1);
        double neg_log2 = (double)precision - log2(p_idx);
        long scaled = (long)(neg_log2 * (double)scale_out + 0.5);
        mvals[tid] = long_to_scalar(scaled);
    }
}

static FrTensor make_log_mvals(uint precision, uint scale_out) {
    uint len = 1u << precision;
    FrTensor mvals(len);
    zklog_init_mvals_kernel<<<(len + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        mvals.gpu_data, precision, scale_out, len);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("zklog_init_mvals_kernel: ") + cudaGetErrorString(err));
    return mvals;
}

// ── Class implementation ──────────────────────────────────────────────────────

zkLog::zkLog(uint precision, uint scale_out)
    : precision(precision), scale_out(scale_out),
      lookup(/*low=*/1, /*len=*/1u << precision, make_log_mvals(precision, scale_out))
{}

pair<FrTensor, FrTensor> zkLog::compute(const FrTensor& probs) {
    return lookup(probs);
}

Fr_t zkLog::prove(const FrTensor& probs, const FrTensor& log_probs, const FrTensor& m,
                  const Fr_t& r, const Fr_t& alpha, const Fr_t& beta,
                  const vector<Fr_t>& u, const vector<Fr_t>& v,
                  const Commitment& sc_pp,
                  vector<Polynomial>& proof, vector<ZKSumcheckProof>& zk_sumchecks) {
    return lookup.prove(probs, log_probs, m, r, alpha, beta, u, v, sc_pp, proof, zk_sumchecks);
}
