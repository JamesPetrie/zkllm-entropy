#include "tensor/fr-tensor.cuh"
#include "util/ioutils.cuh"

using namespace std;

ostream& operator<<(ostream& os, const Fr_t& x)
{
#ifdef USE_GOLDILOCKS
  os << "0x" << std::hex << std::setfill('0') << std::setw(16) << x.val;
  return os << std::dec << std::setw(0) << std::setfill(' ');
#else
  os << "0x" << std::hex;
  for (uint i = 8; i > 0; -- i)
  {
    os << std::setfill('0') << std::setw(8) << x.val[i - 1];
  }
  return os << std::dec << std::setw(0) << std::setfill(' ');
#endif
}

vector<Fr_t> random_vec(uint len)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
    vector<Fr_t> out(len);
#ifdef USE_GOLDILOCKS
    for (uint i = 0; i < len; ++ i) {
        uint64_t lo = dist(mt), hi = dist(mt);
        out[i] = {(hi << 32 | lo) % GOLDILOCKS_P};
    }
#else
    for (uint i = 0; i < len; ++ i) out[i] = {dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt) % 1944954707};
#endif
    return out;
}

uint ceilLog2(uint num) {
    if (num == 0) return 0;
    
    // Decrease num to handle the case where num is already a power of 2
    num--;

    uint result = 0;
    
    // Keep shifting the number to the right until it becomes zero. 
    // Each shift means the number is halved, which corresponds to 
    // a division by 2 in logarithmic terms.
    while (num > 0) {
        num >>= 1;
        result++;
    }

    return result;
}

template<typename T>
std::vector<T> concatenate(const std::vector<std::vector<T>>& vecs) {
    // First, compute the total size for the result vector.
    size_t totalSize = 0;
    for (const auto& v : vecs) {
        totalSize += v.size();
    }

    // Allocate space for the result vector.
    std::vector<T> result;
    result.reserve(totalSize);

    // Append each vector's contents to the result vector.
    for (const auto& v : vecs) {
        result.insert(result.end(), v.begin(), v.end());
    }

    return result;
}

// specify to the compiler that this function needs to be compiled for Fr_t otherwise it cannot be linked
template std::vector<Fr_t> concatenate(const std::vector<std::vector<Fr_t>>& vecs);

// define the kernels

// Elementwise addition
KERNEL void Fr_elementwise_add(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_add(arr1[gid], arr2[gid]);
}

// Broadcast addition
KERNEL void Fr_broadcast_add(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_add(arr[gid], x);
}

// Elementwise negation
KERNEL void Fr_elementwise_neg(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ZERO, arr[gid]);
}

// Elementwise subtraction
KERNEL void Fr_elementwise_sub(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_sub(arr1[gid], arr2[gid]);
}

// Broadcast subtraction
KERNEL void Fr_broadcast_sub(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_sub(arr[gid], x);
}

// To montegomery form
KERNEL void Fr_elementwise_mont(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_mont(arr[gid]);
}

// From montgomery form
KERNEL void Fr_elementwise_unmont(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_unmont(arr[gid]);
}

// Elementwise montegomery multiplication
KERNEL void Fr_elementwise_mont_mul(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_mul(arr1[gid], arr2[gid]);
}

// Broadcast montegomery multiplication
KERNEL void Fr_broadcast_mont_mul(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_mul(arr[gid], x);
}

// Elementwise montegomery multiplication
KERNEL void Fr_elementwise_mul(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(arr1[gid], arr2[gid]));
}

// Broadcast montegomery multiplication
KERNEL void Fr_broadcast_mul(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(arr[gid], x));
}


// ── GPU memory pool ─────────────────────────────────────────────────────────
// Caches freed GPU allocations by size to avoid repeated cudaMalloc/cudaFree.
// This eliminates the main bottleneck in recursive sumcheck proofs where
// thousands of temporary FrTensors are created and destroyed.

#include <unordered_map>
#include <vector>

static std::unordered_map<uint, std::vector<Fr_t*>> gpu_pool;

static Fr_t* pool_alloc(uint size) {
    auto it = gpu_pool.find(size);
    if (it != gpu_pool.end() && !it->second.empty()) {
        Fr_t* ptr = it->second.back();
        it->second.pop_back();
        return ptr;
    }
    Fr_t* ptr = nullptr;
    cudaMalloc((void**)&ptr, sizeof(Fr_t) * size);
    return ptr;
}

static void pool_free(Fr_t* ptr, uint size) {
    if (ptr) gpu_pool[size].push_back(ptr);
}

// implement the class FrTensor

FrTensor::FrTensor(uint size): size(size), gpu_data(nullptr)
{
    gpu_data = pool_alloc(size);
}

FrTensor::FrTensor(uint size, const Fr_t* cpu_data): size(size), gpu_data(nullptr)
{
    gpu_data = pool_alloc(size);
    cudaMemcpy(gpu_data, cpu_data, sizeof(Fr_t) * size, cudaMemcpyHostToDevice);
}

FrTensor::FrTensor(const FrTensor& t): size(t.size), gpu_data(nullptr)
{
    gpu_data = pool_alloc(size);
    cudaMemcpy(gpu_data, t.gpu_data, sizeof(Fr_t) * size, cudaMemcpyDeviceToDevice);
}

FrTensor::~FrTensor()
{
    pool_free(gpu_data, size);
    gpu_data = nullptr;
}

void FrTensor::save(const string& filename) const
{
    savebin(filename, gpu_data, sizeof(Fr_t) * size);
}

KERNEL void scalar_to_int_kernel(const Fr_t* scalar_ptr, int* int_ptr, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    int_ptr[gid] = scalar_to_int(scalar_ptr[gid]);
}

void FrTensor::save_int(const string& filename) const
{
    int* int_gpu_data;
    cudaMalloc((void **)&int_gpu_data, sizeof(int) * size);
    scalar_to_int_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, int_gpu_data, size);
    cudaDeviceSynchronize();
    savebin(filename, int_gpu_data, sizeof(int) * size);
    cudaFree(int_gpu_data);
}

KERNEL void scalar_to_long_kernel(const Fr_t* scalar_ptr, long* long_ptr, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    long_ptr[gid] = scalar_to_long(scalar_ptr[gid]);
}

void FrTensor::save_long(const string& filename) const
{
    long* long_gpu_data;
    cudaMalloc((void **)&long_gpu_data, sizeof(long) * size);
    scalar_to_long_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, long_gpu_data, size);
    cudaDeviceSynchronize();
    savebin(filename, long_gpu_data, sizeof(long) * size);
    cudaFree(long_gpu_data);
}

FrTensor::FrTensor(const string& filename): size(findsize(filename) / sizeof(Fr_t)), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
    loadbin(filename, gpu_data, sizeof(Fr_t) * size);
}

FrTensor FrTensor::from_int_bin(const string& filename)
{
    auto size = findsize(filename) / sizeof(int);
    FrTensor out(size);
    int* int_gpu_data;
    cudaMalloc((void **)&int_gpu_data, sizeof(int) * size);
    loadbin(filename, int_gpu_data, sizeof(int) * size);
    int_to_scalar_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(int_gpu_data, out.gpu_data, size);
    cudaFree(int_gpu_data);
    return out;
}

FrTensor FrTensor::from_long_bin(const string& filename)
{
    auto size = findsize(filename) / sizeof(long);
    FrTensor out(size);
    long* long_gpu_data;
    cudaMalloc((void **)&long_gpu_data, sizeof(long) * size);
    loadbin(filename, long_gpu_data, sizeof(long) * size);
    long_to_scalar_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(long_gpu_data, out.gpu_data, size);
    cudaFree(long_gpu_data);
    return out;
}

Fr_t FrTensor::operator()(uint idx) const
{
    Fr_t out;
    cudaMemcpy(&out, gpu_data + idx, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    return out;
}

// NOTE: cudaDeviceSynchronize() removed from element-wise GPU ops below.
// Within a single CUDA stream, kernel ordering is guaranteed.  Syncs are
// only needed before CPU reads (e.g. .sum(), cudaMemcpy D→H).

FrTensor FrTensor::operator+(const FrTensor& t) const
{
    if (size != t.size) throw std::runtime_error("Incompatible dimensions");
    FrTensor out(size);
    Fr_elementwise_add<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, out.gpu_data, size);
    return out;
}

FrTensor FrTensor::operator+(const Fr_t& x) const
{
    FrTensor out(size);
    Fr_broadcast_add<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, out.gpu_data, size);
    return out;
}

FrTensor& FrTensor::operator+=(const FrTensor& t)
{
    if (size != t.size) throw std::runtime_error("Incompatible dimensions");
    Fr_elementwise_add<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, gpu_data, size);
    return *this;
}

FrTensor& FrTensor::operator+=(const Fr_t& x)
{
    Fr_broadcast_add<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, gpu_data, size);
    return *this;
}

FrTensor FrTensor::operator-() const
{
    FrTensor out(size);
    Fr_elementwise_neg<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, out.gpu_data, size);
    return out;
}

FrTensor FrTensor::operator-(const FrTensor& t) const
{
    if (size != t.size) throw std::runtime_error("Incompatible dimensions");
    FrTensor out(size);
    Fr_elementwise_sub<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, out.gpu_data, size);
    return out;
}

FrTensor FrTensor::operator-(const Fr_t& x) const
{
    FrTensor out(size);
    Fr_broadcast_sub<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, out.gpu_data, size);
    return out;
}

FrTensor& FrTensor::operator-=(const FrTensor& t)
{
    if (size != t.size) throw std::runtime_error("Incompatible dimensions");
    Fr_elementwise_sub<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, gpu_data, size);
    return *this;
}

FrTensor& FrTensor::operator-=(const Fr_t& x)
{
    Fr_broadcast_sub<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, gpu_data, size);
    return *this;
}

FrTensor& FrTensor::mont()
{
    Fr_elementwise_mont<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, gpu_data, size);
    return *this;
}

FrTensor& FrTensor::unmont()
{
    Fr_elementwise_unmont<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, gpu_data, size);
    return *this;
}

FrTensor FrTensor::operator*(const FrTensor& t) const
{
    if (size != t.size) throw std::runtime_error("Incompatible dimensions");
    FrTensor out(size);
    Fr_elementwise_mul<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, out.gpu_data, size);
    return out;
}

FrTensor FrTensor::operator*(const Fr_t& x) const
{
    FrTensor out(size);
    Fr_broadcast_mul<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, out.gpu_data, size);
    return out;
}

FrTensor& FrTensor::operator*=(const FrTensor& t)
{
    if (size != t.size) throw std::runtime_error("Incompatible dimensions");
    Fr_elementwise_mul<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, gpu_data, size);
    return *this;
}

FrTensor& FrTensor::operator*=(const Fr_t& x)
{
    Fr_broadcast_mul<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, gpu_data, size);
    return *this;
}

FrTensor& FrTensor::operator=(const FrTensor& t)
{
    if (size != t.size) throw std::runtime_error("operator=: Incompatible dimensions");
    cudaMemcpy(gpu_data, t.gpu_data, sizeof(Fr_t) * size, cudaMemcpyDeviceToDevice);
    return *this;
}

KERNEL void Fr_sum_reduction(GLOBAL Fr_t *arr, GLOBAL Fr_t *output, uint n) {
    extern __shared__ Fr_t frsum_sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    // Load input into shared memory
    frsum_sdata[tid] = (i < n) ? arr[i] : blstrs__scalar__Scalar_ZERO;
    if (i + blockDim.x < n) frsum_sdata[tid] = blstrs__scalar__Scalar_add(frsum_sdata[tid], arr[i + blockDim.x]);

    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            frsum_sdata[tid] = blstrs__scalar__Scalar_add(frsum_sdata[tid], frsum_sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to output
    if (tid == 0) output[blockIdx.x] = frsum_sdata[0];
}

Fr_t FrTensor::sum() const
{
    Fr_t *ptr_input, *ptr_output;
    uint curSize = size;
    cudaMalloc((void**)&ptr_input, size * sizeof(Fr_t));
    cudaMalloc((void**)&ptr_output, ((size + 1)/ 2) * sizeof(Fr_t));
    cudaMemcpy(ptr_input, gpu_data, size * sizeof(Fr_t), cudaMemcpyDeviceToDevice);

    while(curSize > 1) {
        uint gridSize = (curSize + FrNumThread - 1) / FrNumThread;
        Fr_sum_reduction<<<gridSize, FrNumThread, FrSharedMemorySize>>>(ptr_input, ptr_output, curSize);
        cudaDeviceSynchronize(); // Ensure kernel completion before proceeding
        
        // Swap pointers. Use the output from this step as the input for the next step.
        Fr_t *temp = ptr_input;
        ptr_input = ptr_output;
        ptr_output = temp;
        
        curSize = gridSize;  // The output size is equivalent to the grid size used in the kernel launch
    }

    Fr_t finalSum;
    cudaMemcpy(&finalSum, ptr_input, sizeof(Fr_t), cudaMemcpyDeviceToHost);

    cudaFree(ptr_input);
    cudaFree(ptr_output);

    return finalSum;
}


Fr_t FrTensor::operator()(const vector<Fr_t>& u) const
{
    uint log_dim = u.size();
    if (size <= ((1 << log_dim) / 2) || size > (1 << log_dim)) throw std::runtime_error("Incompatible dimensions");
    return Fr_me(*this, u.begin(), u.end());
}

KERNEL void random_int_kernel(Fr_t* gpu_data, uint num_bits, uint n, unsigned long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;

    // Initialize the RNG state for this thread.
    curand_init(seed, tid, 0, &state);

    if (tid < n) {
#ifdef USE_GOLDILOCKS
        uint64_t raw = curand(&state) & ((1ULL << num_bits) - 1);
        gpu_data[tid] = {raw};
        gpu_data[tid] = blstrs__scalar__Scalar_sub(gpu_data[tid], Fr_t{1ULL << (num_bits - 1)});
#else
        gpu_data[tid] = {curand(&state) & ((1U << num_bits) - 1), 0, 0, 0, 0, 0, 0, 0};
        gpu_data[tid] = blstrs__scalar__Scalar_sub(gpu_data[tid], {1U << (num_bits - 1), 0, 0, 0, 0, 0, 0, 0});
#endif
    }
}

FrTensor FrTensor::random_int(uint size, uint num_bits)
{
    // Create a random device
    std::random_device rd;

    // Initialize a 64-bit Mersenne Twister random number generator
    // with a seed from the random device
    std::mt19937_64 rng(rd());

    // Define the range for your unsigned long numbers
    std::uniform_int_distribution<unsigned long> distribution(0, ULONG_MAX);

    // Generate a random unsigned long number
    unsigned long seed = distribution(rng);

    FrTensor out(size);
    random_int_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(out.gpu_data, num_bits, size, seed);
    return out;
}

KERNEL void random_kernel(Fr_t* gpu_data, uint n, unsigned long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;

    if (tid > n) return;

    // Initialize the RNG state for this thread.
    curand_init(seed, tid, 0, &state);
#ifdef USE_GOLDILOCKS
    uint64_t lo = curand(&state), hi = curand(&state);
    gpu_data[tid] = {(hi << 32 | lo) % GOLDILOCKS_P};
#else
    gpu_data[tid] = {curand(&state), curand(&state), curand(&state), curand(&state), curand(&state), curand(&state), curand(&state), curand(&state) % 1944954707};
#endif
}

FrTensor FrTensor::random(uint size)
{
    // Create a random device
    std::random_device rd;

    // Initialize a 64-bit Mersenne Twister random number generator
    // with a seed from the random device
    std::mt19937_64 rng(rd());

    // Define the range for your unsigned long numbers
    std::uniform_int_distribution<unsigned long> distribution(0, ULONG_MAX);

    // Generate a random unsigned long number
    unsigned long seed = distribution(rng);

    FrTensor out(size);
    random_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(out.gpu_data, size, seed);
    return out;
}

FrTensor FrTensor::partial_me(const vector<Fr_t>& u, uint window_size) const
{
    if (size <= window_size * (1 << (u.size() - 1))) throw std::runtime_error("partial_me: Incompatible dimensions");
    return Fr_partial_me(*this, u.begin(), u.end(), window_size);
}

KERNEL void Fr_multi_dim_partial_me_step(GLOBAL Fr_t* arr_in, GLOBAL Fr_t *arr_out, Fr_t x, uint other_dims, uint in_cur_dim, uint out_cur_dim, uint window_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= other_dims * out_cur_dim * window_size) return;

    uint ind0 = gid / (out_cur_dim * window_size);
    uint ind1 = (gid / window_size) % out_cur_dim;
    uint ind2 = gid % window_size;
    
    x = blstrs__scalar__Scalar_mont(x);

    uint gid0 = ind0 * in_cur_dim * window_size + (2 * ind1) * window_size + ind2;

    if (2 * ind1 + 1 < in_cur_dim) 
    {
        uint gid1 = ind0 * in_cur_dim * window_size + (2 * ind1 + 1) * window_size + ind2;
        arr_out[gid] = blstrs__scalar__Scalar_add(arr_in[gid0], blstrs__scalar__Scalar_mul(x, blstrs__scalar__Scalar_sub(arr_in[gid1], arr_in[gid0])));
    }
    else 
    {
        arr_out[gid] = blstrs__scalar__Scalar_sub(arr_in[gid0], blstrs__scalar__Scalar_mul(x, arr_in[gid0]));
    }
}

FrTensor Fr_partial_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, uint cur_dim, uint window_size)
{
    if (begin >= end) return t;
    if (t.size % (cur_dim * window_size) != 0) throw std::runtime_error("t.size % (cur_dim * window_size) != 0");
    uint cur_dim_out = (cur_dim + 1) / 2;
    uint other_dims = t.size / (cur_dim * window_size);
    uint out_size = other_dims * cur_dim_out * window_size;
    FrTensor t_new(out_size);
    Fr_multi_dim_partial_me_step<<<(t_new.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(t.gpu_data, t_new.gpu_data, *begin, other_dims, cur_dim, cur_dim_out, window_size);
    return Fr_partial_me(t_new, begin + 1, end, cur_dim_out, window_size);
}

FrTensor FrTensor::partial_me(const vector<Fr_t>& u, uint cur_dim, uint window_size) const
{
    if (cur_dim <= ((1<<u.size()) >> 1)) throw std::runtime_error("cur_dim <= ((1<<u.size()) >> 1)");
    if (cur_dim > (1<<u.size())) throw std::runtime_error("cur_dim > (1<<u.size())");
    return Fr_partial_me(*this, u.begin(), u.end(), cur_dim, window_size);
}

KERNEL void Fr_split_by_window(GLOBAL Fr_t *arr_in, GLOBAL Fr_t *arr0, GLOBAL Fr_t *arr1, uint in_size, uint out_size, uint window_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint window_id = gid / window_size;
    uint idx_in_window = gid % window_size;
    uint gid0 = 2 * window_id * window_size + idx_in_window;
    uint gid1 = (2 * window_id + 1) * window_size + idx_in_window;
    arr0[gid] = (gid0 < in_size) ? arr_in[gid0] : blstrs__scalar__Scalar_ZERO;
    arr1[gid] = (gid1 < in_size) ? arr_in[gid1] : blstrs__scalar__Scalar_ZERO;
}

std::pair<FrTensor, FrTensor> FrTensor::split(uint window_size) const
{
    if (window_size < 1 || window_size >= size) throw std::runtime_error("Invalid window size.");
    uint num_window_in = (size + window_size - 1) / window_size;
    uint num_window_out = (num_window_in + 1) / 2;
    uint out_size = num_window_out * window_size; // TODO: BUGGY
    std::pair<FrTensor, FrTensor> out {out_size, out_size};
    Fr_split_by_window<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, out.first.gpu_data, out.second.gpu_data, size, out_size, window_size);
    return out;
}

// ALERT: CONVERTED TO WORK FOR NON-MONTGOMERY FORM
KERNEL void Fr_me_step(GLOBAL Fr_t *arr_in, GLOBAL Fr_t *arr_out, Fr_t x, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;

    x = blstrs__scalar__Scalar_mont(x);
    if (gid1 < in_size) arr_out[gid] = blstrs__scalar__Scalar_add(arr_in[gid0], blstrs__scalar__Scalar_mul(x, blstrs__scalar__Scalar_sub(arr_in[gid1], arr_in[gid0])));
    else if (gid0 < in_size) arr_out[gid] = blstrs__scalar__Scalar_sub(arr_in[gid0], blstrs__scalar__Scalar_mul(x, arr_in[gid0]));
    else arr_out[gid] = blstrs__scalar__Scalar_ZERO;
}

Fr_t Fr_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end)
{
    FrTensor t_new((t.size + 1) / 2);
    if (begin >= end) return t(0);
    Fr_me_step<<<(t_new.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(t.gpu_data, t_new.gpu_data, *begin, t.size, t_new.size);
    return Fr_me(t_new, begin + 1, end);
}

// ALERT: CONVERTED TO WORK FOR NON-MONTGOMERY FORM
KERNEL void Fr_partial_me_step(GLOBAL Fr_t *arr_in, GLOBAL Fr_t *arr_out, Fr_t x, uint in_size, uint out_size, uint window_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint window_id = gid / window_size;
    uint idx_in_window = gid % window_size;
    uint gid0 = 2 * window_id * window_size + idx_in_window;
    uint gid1 = (2 * window_id + 1) * window_size + idx_in_window;

    x = blstrs__scalar__Scalar_mont(x);
    if (gid1 < in_size) arr_out[gid] = blstrs__scalar__Scalar_add(arr_in[gid0], blstrs__scalar__Scalar_mul(x, blstrs__scalar__Scalar_sub(arr_in[gid1], arr_in[gid0])));
    else if (gid0 < in_size) arr_out[gid] = blstrs__scalar__Scalar_sub(arr_in[gid0], blstrs__scalar__Scalar_mul(x, arr_in[gid0]));
    else arr_out[gid] = blstrs__scalar__Scalar_ZERO;
}





FrTensor Fr_partial_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, uint window_size)
{
    if (begin >= end) return t;
    uint num_windows = (t.size + 2 * window_size - 1) / (2 * window_size);
    uint out_size = window_size * num_windows;
    FrTensor t_new(out_size);
    Fr_partial_me_step<<<(t_new.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(t.gpu_data, t_new.gpu_data, *begin, t.size, t_new.size, window_size);
    return Fr_partial_me(t_new, begin + 1, end, window_size);
}

Fr_t FrTensor::multi_dim_me(const vector<vector<Fr_t>>& us, const vector<uint>& shape) const
{
    if (shape.size() != us.size()) throw std::runtime_error("Incompatible dimensions");
    if (shape.size() == 0) return (*this)(0);
    else if (shape.size() == 1) return (*this)(us[0]);
    else {
        FrTensor t_new = this -> partial_me(us.back(), shape.back(), 1);
        return t_new.multi_dim_me({us.begin(), us.end() - 1}, {shape.begin(), shape.end() - 1});
    }
}

ostream& operator<<(ostream& os, const FrTensor& A)
{
    os << '['; 
    for (uint i = 0; i < A.size - 1; ++ i) os << A(i) << '\n';
    os << A(A.size-1) << ']';
    return os;
}

// Input: x is in montgomery form
// Output: x^{-1} = x^{P-2} in montgomery form
DEVICE Fr_t modular_inverse(Fr_t x){
#ifdef USE_GOLDILOCKS
    return gold_inverse(x);
#else
    Fr_t P_sub2 = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_P, {2,0,0,0,0,0,0,0});

    // for each bit of P_sub2, compute x^i
    Fr_t res = blstrs__scalar__Scalar_ONE;
    for(int i = 0; i < blstrs__scalar__Scalar_LIMBS -1 ; i++){
        uint32_t exponent = P_sub2.val[i];
        for(int j = 0; j < blstrs__scalar__Scalar_LIMB_BITS; j++){
            if(exponent & 1){
                res = blstrs__scalar__Scalar_mul(res, x);
            }
            exponent = exponent >> 1;
            x = blstrs__scalar__Scalar_sqr(x);
        }
    }

    uint32_t exponent = P_sub2.val[blstrs__scalar__Scalar_LIMBS - 1];
    while(exponent > 0){
        if(exponent & 1){
            res = blstrs__scalar__Scalar_mul(res, x);
        }
        exponent = exponent >> 1;
        x = blstrs__scalar__Scalar_sqr(x);
    }
    return res;
#endif
}

#ifdef USE_GOLDILOCKS
DEVICE Fr_t ulong_to_scalar(unsigned long num)
{
    return {num % GOLDILOCKS_P};
}

DEVICE Fr_t long_to_scalar(long num)
{
    if (num >= 0) return {static_cast<uint64_t>(num)};
    else return gold_sub({0ULL}, {static_cast<uint64_t>(-num)});
}

DEVICE Fr_t uint_to_scalar(uint num)
{
    return {static_cast<uint64_t>(num)};
}

DEVICE Fr_t int_to_scalar(int num)
{
    if (num >= 0) return {static_cast<uint64_t>(num)};
    else return gold_sub({0ULL}, {static_cast<uint64_t>(-num)});
}
#else
DEVICE Fr_t ulong_to_scalar(unsigned long num)
{
    return {static_cast<uint>(num), static_cast<uint>(num >> 32), 0, 0, 0, 0, 0, 0};
}

DEVICE Fr_t long_to_scalar(long num)
{
    if (num >= 0) return ulong_to_scalar(static_cast<unsigned long>(num));
    else return blstrs__scalar__Scalar_sub({0,0,0,0,0,0,0,0}, ulong_to_scalar(static_cast<unsigned long>(-num)));
}

DEVICE Fr_t uint_to_scalar(uint num)
{
    return {num, 0, 0, 0, 0, 0, 0, 0};
}

DEVICE Fr_t int_to_scalar(int num)
{
    if (num >= 0) return uint_to_scalar(static_cast<uint>(num));
    else return blstrs__scalar__Scalar_sub({0,0,0,0,0,0,0,0}, uint_to_scalar(static_cast<uint>(-num)));
}
#endif

KERNEL void int_to_scalar_kernel(int* int_ptr, Fr_t* scalar_ptr, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    scalar_ptr[gid] = int_to_scalar(int_ptr[gid]);
}

KERNEL void long_to_scalar_kernel(long* long_ptr, Fr_t* scalar_ptr, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    scalar_ptr[gid] = long_to_scalar(long_ptr[gid]);
}

FrTensor::FrTensor(uint size, const int* cpu_data): size(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
    int* int_gpu_data;
    cudaMalloc((void **)&int_gpu_data, sizeof(int) * size);
    cudaMemcpy(int_gpu_data, cpu_data, sizeof(int) * size, cudaMemcpyHostToDevice);
    int_to_scalar_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(int_gpu_data, gpu_data, size);
    cudaDeviceSynchronize();
    cudaFree(int_gpu_data);
}

FrTensor::FrTensor(uint size, const long* cpu_data): size(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
    long* long_gpu_data;
    cudaMalloc((void **)&long_gpu_data, sizeof(long) * size);
    cudaMemcpy(long_gpu_data, cpu_data, sizeof(long) * size, cudaMemcpyHostToDevice);
    long_to_scalar_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(long_gpu_data, gpu_data, size);
    cudaDeviceSynchronize();
    cudaFree(long_gpu_data);
}

DEVICE Fr_t float_to_scalar(float x, unsigned long scaling_factor)
{
    x = x * scaling_factor;
    if (x >= 0) return long_to_scalar(static_cast<long>(x + 0.5));
    else return long_to_scalar(static_cast<long>(x - 0.5));
}

KERNEL void float_to_scalar_kernel(float* float_ptr, Fr_t* scalar_ptr, unsigned long scaling_factor, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    scalar_ptr[gid] = float_to_scalar(float_ptr[gid], scaling_factor);
}

DEVICE Fr_t double_to_scalar(double x, unsigned long scaling_factor)
{
    x = x * scaling_factor;
    if (x >= 0) return long_to_scalar(static_cast<long>(x + 0.5));
    else return long_to_scalar(static_cast<long>(x - 0.5));
}

KERNEL void double_to_scalar_kernel(double* double_ptr, Fr_t* scalar_ptr, unsigned long scaling_factor, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    scalar_ptr[gid] = double_to_scalar(double_ptr[gid], scaling_factor);
}

FrTensor::FrTensor(uint size, const float* cpu_data, unsigned long scaling_factor): size(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
    float* float_gpu_data;
    cudaMalloc((void **)&float_gpu_data, sizeof(float) * size);
    cudaMemcpy(float_gpu_data, cpu_data, sizeof(float) * size, cudaMemcpyHostToDevice);
    float_to_scalar_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(float_gpu_data, gpu_data, scaling_factor, size);
    cudaDeviceSynchronize();
    cudaFree(float_gpu_data);

}

FrTensor::FrTensor(uint size, const double* cpu_data, unsigned long scaling_factor): size(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
    double* double_gpu_data;
    cudaMalloc((void **)&double_gpu_data, sizeof(double) * size);
    cudaMemcpy(double_gpu_data, cpu_data, sizeof(double) * size, cudaMemcpyHostToDevice);
    double_to_scalar_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(double_gpu_data, gpu_data, scaling_factor, size);
    cudaDeviceSynchronize();
    cudaFree(double_gpu_data);
}

KERNEL void FrTensor_pad_kernel(GLOBAL Fr_t* arr_in, GLOBAL Fr_t* arr_out, uint N, uint last_dim_in, uint last_dim_out, Fr_t pad_val)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= N) return;
    auto gid0 = gid / last_dim_out, gid1 = gid % last_dim_out;
    if (gid1 >= last_dim_in) arr_out[gid] = pad_val;
    else arr_out[gid] = arr_in[gid0 * last_dim_in + gid1];
}

FrTensor FrTensor::pad(const vector<uint>& shape, const Fr_t& pad_val) const
{
    uint cum_shape = 1;
    for (auto& s: shape) cum_shape *= s;
    if (cum_shape != size) throw std::runtime_error("pad: cum_shape != size");

    uint last_dim = shape.back();
    uint last_dim_padded = 1 << ceilLog2(last_dim);

    FrTensor out((cum_shape / last_dim) * last_dim_padded);
    FrTensor_pad_kernel<<<(out.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, out.gpu_data, out.size, last_dim, last_dim_padded, pad_val);

    if (shape.size() == 1) return out;
    else {
        vector<uint> shape_new(shape.begin(), shape.end() - 1);
        shape_new.back() *= last_dim_padded;
        return out.pad(shape_new, pad_val);
    }
}




KERNEL void matrixMultiplyOptimized(Fr_t* A, Fr_t* B, Fr_t* C, int rowsA, int colsA, int colsB) {
    __shared__ Fr_t A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ Fr_t B_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    Fr_t sum = blstrs__scalar__Scalar_ZERO;

    for (int t = 0; t < (colsA - 1)/TILE_WIDTH + 1; ++t) {
        if (row < rowsA && t*TILE_WIDTH + threadIdx.x < colsA) {
            A_tile[threadIdx.y][threadIdx.x] = A[row*colsA + t*TILE_WIDTH + threadIdx.x];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;
        }

        if (t*TILE_WIDTH + threadIdx.y < colsA && col < colsB) {
            B_tile[threadIdx.y][threadIdx.x] = B[(t*TILE_WIDTH + threadIdx.y)*colsB + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum = blstrs__scalar__Scalar_add(sum, blstrs__scalar__Scalar_mul(A_tile[threadIdx.y][k], B_tile[k][threadIdx.x]));
        }

        __syncthreads();
    }

    if (row < rowsA && col < colsB) {
        C[row*colsB + col] = blstrs__scalar__Scalar_mont(sum);
    }
}


#ifdef USE_GOLDILOCKS
// ── Optimized Goldilocks matmul with lazy accumulation ──────────────────────
//
// Key optimization: replace 15 full gold_add calls (each ~10 SASS instructions
// with modular reduction) with raw uint64_t wrapping additions + a single
// final reduction per tile pass.
//
// The kernel is throughput-limited (total SASS instruction count), not
// latency-limited (dependency chains). With 16 warps per scheduler, ILP is
// fully utilized. So the only way to speed up is fewer instructions.
//
// Correctness: each gold_mul result is in [0, p) where p = 2^64 - 2^32 + 1.
// Summing 16 such values in wrapping uint64_t arithmetic gives us
// (true_sum mod 2^64). We track the number of 2^64 overflows in a counter.
// Then: true_sum ≡ acc_lo + overflow_count * 2^64 ≡ acc_lo + overflow_count * eps (mod p)
// where eps = 2^32 - 1.

KERNEL void matrixMultiplyGoldV2(Fr_t* A, Fr_t* B, Fr_t* C,
                                  int rowsA, int colsA, int colsB) {
    __shared__ Fr_t A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ Fr_t B_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int numTiles = (colsA - 1) / TILE_WIDTH + 1;

    // Running sum in fully-reduced form [0, p)
    uint64_t sum = 0;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile
        if (row < rowsA && t * TILE_WIDTH + tx < colsA) {
            A_tile[ty][tx] = A[row * colsA + t * TILE_WIDTH + tx];
        } else {
            A_tile[ty][tx] = Fr_t{0ULL};
        }
        if (t * TILE_WIDTH + ty < colsA && col < colsB) {
            B_tile[ty][tx] = B[(t * TILE_WIDTH + ty) * colsB + col];
        } else {
            B_tile[ty][tx] = Fr_t{0ULL};
        }
        __syncthreads();

        // Accumulate 16 products using wrapping uint64_t addition.
        // Track overflow count for final Goldilocks reduction.
        uint64_t acc = 0;
        uint32_t overflows = 0;

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Fr_t prod = gold_mul(A_tile[ty][k], B_tile[k][tx]);
            uint64_t new_acc = acc + prod.val;
            if (new_acc < acc) overflows++;  // wrapped around 2^64
            acc = new_acc;
        }

        // Reduce: true_sum = acc + overflows * 2^64
        // 2^64 ≡ eps (mod p) where eps = GOLDILOCKS_P_NEG = 2^32 - 1
        // overflows <= 15 (at most 16 additions of values < 2^64)
        // overflows * eps < 16 * 2^32 = 2^36, fits comfortably in uint64_t
        uint64_t correction = (uint64_t)overflows * GOLDILOCKS_P_NEG;
        uint64_t tile_sum = acc + correction;
        if (tile_sum < acc) {
            // This addition itself overflowed: add another eps
            tile_sum += GOLDILOCKS_P_NEG;
        }
        // Canonical reduction
        if (tile_sum >= GOLDILOCKS_P) tile_sum -= GOLDILOCKS_P;
        // tile_sum might still be >= p if the correction pushed it over
        if (tile_sum >= GOLDILOCKS_P) tile_sum -= GOLDILOCKS_P;

        // Accumulate tile result into running sum
        uint64_t new_sum = sum + tile_sum;
        // Check for overflow and reduce
        if (new_sum < sum || new_sum >= GOLDILOCKS_P) {
            new_sum -= GOLDILOCKS_P;
        }
        sum = new_sum;

        __syncthreads();
    }

    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = Fr_t{sum};
    }
}
// ── V3: Skip per-multiply reduction, accumulate raw 128-bit products ─────────
//
// gold_mul spends ~18 SASS instructions on Plonky2 reduction per call.
// With 16 calls per tile, that's ~288 SASS just for reductions.
// V3 computes only the 128-bit product (lo, hi) and accumulates them separately:
//   sum_lo += lo, sum_hi += hi  (with overflow counters)
// Then does ONE Plonky2 reduction per tile pass.
//
// Correctness: each product a*b where a,b < p < 2^64 yields (lo, hi) with
// true value = hi * 2^64 + lo. Accumulating 16 such products:
//   total = sum(hi_k) * 2^64 + sum(lo_k)
// We track sum_lo, sum_hi as wrapping uint64_t with overflow counters.
// Final: total = (ov_hi * 2^64 + sum_hi) * 2^64 + (ov_lo * 2^64 + sum_lo)
//              = ov_hi * 2^128 + (sum_hi + ov_lo) * 2^64 + sum_lo
// Reduce mod p using: 2^64 ≡ eps, 2^128 ≡ -2^32 (mod p).

KERNEL void matrixMultiplyGoldV3(Fr_t* A, Fr_t* B, Fr_t* C,
                                  int rowsA, int colsA, int colsB) {
    __shared__ Fr_t A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ Fr_t B_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int numTiles = (colsA - 1) / TILE_WIDTH + 1;

    // Running sum in fully-reduced form [0, p)
    uint64_t sum = 0;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile
        if (row < rowsA && t * TILE_WIDTH + tx < colsA) {
            A_tile[ty][tx] = A[row * colsA + t * TILE_WIDTH + tx];
        } else {
            A_tile[ty][tx] = Fr_t{0ULL};
        }
        if (t * TILE_WIDTH + ty < colsA && col < colsB) {
            B_tile[ty][tx] = B[(t * TILE_WIDTH + ty) * colsB + col];
        } else {
            B_tile[ty][tx] = Fr_t{0ULL};
        }
        __syncthreads();

        // Accumulate raw 128-bit products without per-multiply reduction
        uint64_t tile_lo = 0, tile_hi = 0;
        uint32_t ov_lo = 0, ov_hi = 0;

        for (int k = 0; k < TILE_WIDTH; ++k) {
            uint64_t a = A_tile[ty][k].val;
            uint64_t b = B_tile[k][tx].val;

            // 128-bit product only — no Plonky2 reduction
            uint64_t lo = a * b;
            uint64_t hi = UMUL64HI(a, b);

            // Accumulate lo
            uint64_t new_lo = tile_lo + lo;
            if (new_lo < tile_lo) ov_lo++;
            tile_lo = new_lo;

            // Accumulate hi
            uint64_t new_hi = tile_hi + hi;
            if (new_hi < tile_hi) ov_hi++;
            tile_hi = new_hi;
        }

        // ── Single reduction per tile ──────────────────────────────────
        // total = (ov_hi * 2^64 + tile_hi) * 2^64 + (ov_lo * 2^64 + tile_lo)
        //       = ov_hi * 2^128 + (tile_hi + ov_lo) * 2^64 + tile_lo
        //
        // Modular equivalences (p = 2^64 - 2^32 + 1, eps = 2^32 - 1):
        //   2^64  ≡ eps (mod p)
        //   2^128 ≡ eps^2 = 2^64 - 2^33 + 1 ≡ eps - 2^33 + 1 = -(2^32) (mod p)
        //
        // result ≡ tile_lo + (tile_hi + ov_lo) * eps - ov_hi * 2^32 (mod p)

        // Step 1: Plonky2-reduce the 128-bit value (tile_lo, tile_hi)
        // This is the same reduction as gold_mul but applied to accumulated sums
        uint32_t hi_hi = (uint32_t)(tile_hi >> 32);
        uint32_t hi_lo = (uint32_t)tile_hi;

        uint64_t t0 = tile_lo - (uint64_t)hi_hi;
        if (tile_lo < (uint64_t)hi_hi) t0 -= GOLDILOCKS_P_NEG;

        uint64_t t1 = (uint64_t)hi_lo * GOLDILOCKS_P_NEG;

        uint64_t result = t0 + t1;
        if (result < t0) result += GOLDILOCKS_P_NEG;
        if (result >= GOLDILOCKS_P) result -= GOLDILOCKS_P;

        // Step 2: Add ov_lo * eps correction
        // ov_lo <= 15, eps < 2^32, so ov_lo * eps < 2^36 — fits in uint64_t
        if (ov_lo > 0) {
            uint64_t lo_corr = (uint64_t)ov_lo * GOLDILOCKS_P_NEG;
            uint64_t r2 = result + lo_corr;
            if (r2 < result) r2 += GOLDILOCKS_P_NEG;  // overflow: add eps
            if (r2 >= GOLDILOCKS_P) r2 -= GOLDILOCKS_P;
            result = r2;
        }

        // Step 3: Subtract ov_hi * 2^32 correction
        // ov_hi <= 15, so ov_hi * 2^32 < 2^36 — fits in uint64_t
        if (ov_hi > 0) {
            uint64_t hi_corr = (uint64_t)ov_hi << 32;
            if (result >= hi_corr) {
                result -= hi_corr;
            } else {
                result = result + GOLDILOCKS_P - hi_corr;
            }
            if (result >= GOLDILOCKS_P) result -= GOLDILOCKS_P;
        }

        // Accumulate tile result into running sum
        uint64_t new_sum = sum + result;
        if (new_sum < sum || new_sum >= GOLDILOCKS_P) {
            new_sum -= GOLDILOCKS_P;
        }
        sum = new_sum;

        __syncthreads();
    }

    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = Fr_t{sum};
    }
}
#endif // USE_GOLDILOCKS


FrTensor FrTensor::matmul(const FrTensor& x, const FrTensor& y, uint M, uint N, uint P)
{
    if (x.size != M * N || y.size != N * P) throw std::runtime_error("matmul: incompatible dimensions");
    FrTensor out(M * P);
#ifdef USE_GOLDILOCKS
    matrixMultiplyGoldV3<<<dim3((P-1)/TILE_WIDTH + 1, (M-1)/TILE_WIDTH + 1), dim3(TILE_WIDTH, TILE_WIDTH)>>>(x.gpu_data, y.gpu_data, out.gpu_data, M, N, P);
#else
    matrixMultiplyOptimized<<<dim3((P-1)/TILE_WIDTH + 1, (M-1)/TILE_WIDTH + 1), dim3(TILE_WIDTH, TILE_WIDTH)>>>(x.gpu_data, y.gpu_data, out.gpu_data, M, N, P);
#endif
    return out;
}




#ifdef USE_GOLDILOCKS
// ── Matmul kernel with fp16 weight storage ──────────────────────────────────
// Weights are stored as fp16 (2 bytes) and converted to Goldilocks field
// elements on the fly during the B-tile load.  This reduces weight memory
// by 4x and improves L2 cache hit rate for 7B-scale weight matrices.
//
// The conversion is: fp16 -> float -> * scaling_factor -> round -> long_to_scalar
// which reproduces the exact same field elements as the offline quantization.

__device__ inline Fr_t fp16_to_field(__half w, unsigned long scaling_factor) {
    float f = __half2float(w);
    long q = __float2ll_rn(f * (float)scaling_factor);
    return long_to_scalar(q);
}

KERNEL void matmul_fp16w(const Fr_t* A, const __half* B_fp16, Fr_t* C,
                         int rowsA, int colsA, int colsB,
                         unsigned long scaling_factor)
{
    __shared__ Fr_t A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ Fr_t B_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    Fr_t sum = blstrs__scalar__Scalar_ZERO;

    for (int t = 0; t < (colsA - 1)/TILE_WIDTH + 1; ++t) {
        // Load A tile (already field elements)
        if (row < rowsA && t*TILE_WIDTH + threadIdx.x < colsA)
            A_tile[threadIdx.y][threadIdx.x] = A[row*colsA + t*TILE_WIDTH + threadIdx.x];
        else
            A_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;

        // Load B tile: fp16 -> field element conversion in registers
        if (t*TILE_WIDTH + threadIdx.y < colsA && col < colsB) {
            __half w = B_fp16[(t*TILE_WIDTH + threadIdx.y)*colsB + col];
            B_tile[threadIdx.y][threadIdx.x] = fp16_to_field(w, scaling_factor);
        } else {
            B_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum = blstrs__scalar__Scalar_add(sum,
                  blstrs__scalar__Scalar_mul(A_tile[threadIdx.y][k], B_tile[k][threadIdx.x]));
        }

        __syncthreads();
    }

    if (row < rowsA && col < colsB)
        C[row*colsB + col] = blstrs__scalar__Scalar_mont(sum);
}

FrTensor FrTensor::matmul_fp16w(const FrTensor& activations, const __half* weights_fp16,
                                uint M, uint N, uint P, unsigned long scaling_factor)
{
    if (activations.size != M * N) throw std::runtime_error("matmul_fp16w: incompatible activation dimensions");
    FrTensor out(M * P);
    ::matmul_fp16w<<<dim3((P-1)/TILE_WIDTH + 1, (M-1)/TILE_WIDTH + 1), dim3(TILE_WIDTH, TILE_WIDTH)>>>(
        activations.gpu_data, weights_fp16, out.gpu_data, M, N, P, scaling_factor);
    return out;
}
#endif

// implement a kernel to transpose a matrix of size M by N
KERNEL void transpose_kernel(Fr_t* in_ptr, Fr_t* out_ptr, int M, int N) {
    uint gid = GET_GLOBAL_ID();
    if (gid >= M*N) return;
    int row = gid / N;
    int col = gid % N;
    out_ptr[col*M + row] = in_ptr[row*N + col];
}

FrTensor FrTensor::transpose(uint M, uint N) const
{
    if (size != M * N) throw std::runtime_error("transpose: incompatible dimensions");
    FrTensor out(N * M);
    transpose_kernel<<<(M*N+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, out.gpu_data, M, N);
    return out;
}

FrTensor catTensors(const vector<FrTensor>& vec){
    // make sure all tensors have the same size
    uint size = vec[0].size;
    for (uint i = 1; i < vec.size(); i++){
        if (vec[i].size != size){  
            cout << "tensor size does not match: vec[0].size = "<<size<<", vec["<<i<<"].size = "<<vec[i].size<<"!" << endl;
            throw std::runtime_error("tensor size does not match!");
        }
    }
    uint num = vec.size();
    FrTensor out(num * size);
    
    // copy data to out
    for (uint i = 0; i < num; i++){
        cudaMemcpy(out.gpu_data + i * size, vec[i].gpu_data, size * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    }

    return out;
}

#ifdef USE_GOLDILOCKS
DEVICE unsigned int scalar_to_uint(Fr_t x)
{
    if (x.val <= 0xFFFFFFFFULL) return static_cast<unsigned int>(x.val);
    return 0U;
}

DEVICE int scalar_to_int(Fr_t x)
{
    // Positive: val < 2^31
    if (x.val < 0x80000000ULL) return static_cast<int>(x.val);
    // Negative: val > p/2 means it represents a negative number
    return -static_cast<int>(gold_sub({0ULL}, x).val);
}

DEVICE unsigned long scalar_to_ulong(Fr_t x)
{
    return x.val;
}

DEVICE long scalar_to_long(Fr_t x)
{
    // Positive: val < p/2
    if (x.val <= (GOLDILOCKS_P >> 1)) return static_cast<long>(x.val);
    // Negative
    return -static_cast<long>(gold_sub({0ULL}, x).val);
}
#else
DEVICE unsigned int scalar_to_uint(Fr_t x)
{
    if (!x.val[7] && !x.val[6] && !x.val[5] && !x.val[4] && !x.val[3] && !x.val[2] && !x.val[1]) return static_cast<unsigned int>(x.val[0]);
    return 0U;

}

DEVICE int scalar_to_int(Fr_t x)
{
    if (!x.val[7] && !x.val[6] && !x.val[5] && !x.val[4] && !x.val[3] && !x.val[2] && !x.val[1] && !(x.val[0] >> 31)) return static_cast<int>(scalar_to_uint(x));
    else return -static_cast<int>(scalar_to_uint(blstrs__scalar__Scalar_sub({0,0,0,0,0,0,0,0}, x)));
}

DEVICE unsigned long scalar_to_ulong(Fr_t x)
{
    if (!x.val[7] && !x.val[6] && !x.val[5] && !x.val[4] && !x.val[3] && !x.val[2]) return static_cast<unsigned long>(x.val[0]) | (static_cast<unsigned long>(x.val[1]) << 32);
    return 0UL;
}

DEVICE long scalar_to_long(Fr_t x)
{
    if (!x.val[7] && !x.val[6] && !x.val[5] && !x.val[4] && !x.val[3] && !x.val[2] && !(x.val[1] >> 31)) return static_cast<long>(scalar_to_ulong(x));
    else return -static_cast<long>(scalar_to_ulong(blstrs__scalar__Scalar_sub({0,0,0,0,0,0,0,0}, x)));
}
#endif

FrTensor FrTensor::trunc(uint begin_idx, uint end_idx) const
{
    if (begin_idx >= end_idx || end_idx > size) throw std::runtime_error("trunc: invalid indices");
    FrTensor out(end_idx - begin_idx);
    cudaMemcpy(out.gpu_data, gpu_data + begin_idx, out.size * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    return out;
}