#ifndef RESCALING_CUH
#define RESCALING_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include "tensor/fr-tensor.cuh"
#include "zknn/tlookup.cuh"
#include "proof/proof.cuh"

class Rescaling {
public:
    uint scaling_factor;
    tLookupRange tl_rem; // table for remainder
    Rescaling decomp(const FrTensor& X, FrTensor& rem);
    FrTensor *rem_tensor_ptr;

    Rescaling(uint scaling_factor);
    FrTensor operator()(const FrTensor& X);
    vector<Claim> prove(const FrTensor& X, const FrTensor& X_);
    ~Rescaling();
};

#endif // RESCALING_CUH