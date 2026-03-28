#include "zknn/zksoftmax.cuh"
#include "zknn/zkfc.cuh"
#include "tensor/fr-tensor.cuh"
#include "proof/proof.cuh"
#ifndef USE_GOLDILOCKS
#include "commit/commitment.cuh"
#endif
#include "zknn/rescaling.cuh"
#include <string>

int main(int argc, char *argv[])
{
    string block_input_fn = argv[1];
    string block_output_fn = argv[2];
    string output_fn = argv[3];

    FrTensor x = FrTensor::from_int_bin(block_input_fn);
    FrTensor y = FrTensor::from_int_bin(block_output_fn);
    FrTensor z = x + y;
    z.save_int(output_fn);

    // cout << O_(0) << " " << O_(O_.size - 1) << endl;
    return 0;
}