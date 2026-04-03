#include "poly/polynomial.cuh"

#ifdef USE_GOLDILOCKS
// ── CPU-side scalar field operators (Goldilocks) ─────────────────────────────
// Goldilocks arithmetic is simple 64-bit integer math — no need for GPU kernels.
// This eliminates ~16k cudaMalloc/cudaFree calls per layer proof.

Fr_t operator+(const Fr_t& a, const Fr_t& b) {
    return gold_add(a, b);
}

Fr_t operator-(const Fr_t& a, const Fr_t& b) {
    return gold_sub(a, b);
}

Fr_t operator-(const Fr_t& a) {
    return gold_sub(Gold_t{0ULL}, a);
}

Fr_t operator*(const Fr_t& a, const Fr_t& b) {
    return gold_mul(a, b);
}

Fr_t operator/(const Fr_t& a, const Fr_t& b) {
    if (b.val == 0) throw std::runtime_error("divide by zero");
    return gold_div(a, b);
}

Fr_t inv(const Fr_t& a) {
    if (a.val == 0) throw std::runtime_error("inverse of zero");
    return gold_inverse(a);
}

#else
//kernel for operator+
__global__ void addKernel(const Fr_t* a, const Fr_t* b, Fr_t* c)
{
    *c = blstrs__scalar__Scalar_add(*a, *b);
}

Fr_t operator+(const Fr_t& a, const Fr_t& b)
{
    //copy a and b to cuda
    Fr_t* a_cuda, *b_cuda, *c_cuda;
    cudaMalloc((void**)&a_cuda, sizeof(Fr_t));
    cudaMalloc((void**)&b_cuda, sizeof(Fr_t));
    cudaMalloc((void**)&c_cuda, sizeof(Fr_t));
    cudaMemcpy(a_cuda, &a, sizeof(Fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, &b, sizeof(Fr_t), cudaMemcpyHostToDevice);
    addKernel<<<1, 1>>>(a_cuda, b_cuda, c_cuda);
    Fr_t c;
    cudaMemcpy(&c, c_cuda, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);
    return c;
}

//kernel for operator-
__global__ void subKernel(const Fr_t* a, const Fr_t* b, Fr_t* c)
{
    *c = blstrs__scalar__Scalar_sub(*a, *b);
}

Fr_t operator-(const Fr_t& a, const Fr_t& b)
{
    //copy a and b to cuda
    Fr_t* a_cuda, *b_cuda, *c_cuda;
    cudaMalloc((void**)&a_cuda, sizeof(Fr_t));
    cudaMalloc((void**)&b_cuda, sizeof(Fr_t));
    cudaMalloc((void**)&c_cuda, sizeof(Fr_t));
    cudaMemcpy(a_cuda, &a, sizeof(Fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, &b, sizeof(Fr_t), cudaMemcpyHostToDevice);
    subKernel<<<1, 1>>>(a_cuda, b_cuda, c_cuda);
    Fr_t c;
    cudaMemcpy(&c, c_cuda, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);
    return c;
}

__global__ void negKernel(const Fr_t* a, Fr_t* c)
{
    *c = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ZERO, *a);
}

Fr_t operator-(const Fr_t& a)
{
    //copy a to cuda
    Fr_t* a_cuda, *c_cuda;
    cudaMalloc((void**)&a_cuda, sizeof(Fr_t));
    cudaMalloc((void**)&c_cuda, sizeof(Fr_t));
    cudaMemcpy(a_cuda, &a, sizeof(Fr_t), cudaMemcpyHostToDevice);
    negKernel<<<1, 1>>>(a_cuda, c_cuda);
    Fr_t c;
    cudaMemcpy(&c, c_cuda, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(a_cuda);
    cudaFree(c_cuda);
    return c;
}

__global__ void mulKernel(const Fr_t* a, const Fr_t* b, Fr_t* c)
{
    *c = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(*a, *b));
}

Fr_t operator*(const Fr_t& a, const Fr_t& b)
{
    //copy a and b to cuda
    Fr_t* a_cuda, *b_cuda, *c_cuda;
    cudaMalloc((void**)&a_cuda, sizeof(Fr_t));
    cudaMalloc((void**)&b_cuda, sizeof(Fr_t));
    cudaMalloc((void**)&c_cuda, sizeof(Fr_t));
    cudaMemcpy(a_cuda, &a, sizeof(Fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, &b, sizeof(Fr_t), cudaMemcpyHostToDevice);
    mulKernel<<<1, 1>>>(a_cuda, b_cuda, c_cuda);
    Fr_t c;
    cudaMemcpy(&c, c_cuda, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);
    return c;
}

__global__ void divKernel(const Fr_t* a, const Fr_t* b, Fr_t* c)
{
    auto a_mont = blstrs__scalar__Scalar_mont(*a);
    auto b_mont = blstrs__scalar__Scalar_mont(*b);
    *c = blstrs__scalar__Scalar_unmont(blstrs__scalar__Scalar_div(a_mont, b_mont));
}

Fr_t operator/(const Fr_t& a, const Fr_t& b)
{
#ifdef USE_GOLDILOCKS
    if (b.val == 0) {
#else
    if (!b.val[0] && !b.val[1] && !b.val[2] && !b.val[3] && !b.val[4] && !b.val[5] && !b.val[6] && !b.val[7]) {
#endif
        throw std::runtime_error("divide by zero");
    }
    //copy a and b to cuda
    Fr_t* a_cuda, *b_cuda, *c_cuda;
    cudaMalloc((void**)&a_cuda, sizeof(Fr_t));
    cudaMalloc((void**)&b_cuda, sizeof(Fr_t));
    cudaMalloc((void**)&c_cuda, sizeof(Fr_t));
    cudaMemcpy(a_cuda, &a, sizeof(Fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, &b, sizeof(Fr_t), cudaMemcpyHostToDevice);
    divKernel<<<1, 1>>>(a_cuda, b_cuda, c_cuda);
    Fr_t c;
    cudaMemcpy(&c, c_cuda, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);
    return c;
}

__global__ void invKernel(const Fr_t* a, Fr_t* c)
{   
    Fr_t a_mont = blstrs__scalar__Scalar_mont(*a);
    *c = blstrs__scalar__Scalar_unmont(blstrs__scalar__Scalar_inverse(a_mont));
}

Fr_t inv(const Fr_t& a)
{   
#ifdef USE_GOLDILOCKS
    if (a.val == 0) {
#else
    if (!a.val[0] && !a.val[1] && !a.val[2] && !a.val[3] && !a.val[4] && !a.val[5] && !a.val[6] && !a.val[7]) {
#endif
        throw std::runtime_error("divide by zero");
    }
    //copy a to cuda
    Fr_t* a_cuda, *c_cuda;
    cudaMalloc((void**)&a_cuda, sizeof(Fr_t));
    cudaMalloc((void**)&c_cuda, sizeof(Fr_t));
    cudaMemcpy(a_cuda, &a, sizeof(Fr_t), cudaMemcpyHostToDevice);
    invKernel<<<1, 1>>>(a_cuda, c_cuda);
    Fr_t c;
    cudaMemcpy(&c, c_cuda, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(a_cuda);
    cudaFree(c_cuda);
    return c;
}

#endif

Polynomial::Polynomial() : degree_(0), coefficients_(nullptr) {}

Polynomial::Polynomial(int degree) : degree_(degree) {
    cudaMalloc((void**)&coefficients_, (degree + 1) * sizeof(Fr_t));
    cudaMemset(coefficients_, 0, (degree + 1) * sizeof(Fr_t));
}

Polynomial::Polynomial(int degree, Fr_t* coefficients) : degree_(degree) {
    cudaMalloc((void**)&coefficients_, (degree + 1) * sizeof(Fr_t));
    cudaMemcpy(coefficients_, coefficients, (degree + 1) * sizeof(Fr_t), cudaMemcpyHostToDevice);
}

Polynomial::Polynomial(const Polynomial& other) : degree_(other.degree_) {
    cudaMalloc((void**)&coefficients_, (degree_ + 1) * sizeof(Fr_t));
    cudaMemcpy(coefficients_, other.coefficients_, (degree_ + 1) * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
}

Polynomial::Polynomial(const Fr_t& constant) : degree_(0) {
    cudaMalloc((void**)&coefficients_, sizeof(Fr_t));
    cudaMemcpy(coefficients_, &constant, sizeof(Fr_t), cudaMemcpyHostToDevice);
}

Polynomial::Polynomial(const vector<Fr_t>& coefficients) : degree_(coefficients.size() - 1) {
    cudaMalloc((void**)&coefficients_, (degree_ + 1) * sizeof(Fr_t));
    cudaMemcpy(coefficients_, coefficients.data(), (degree_ + 1) * sizeof(Fr_t), cudaMemcpyHostToDevice);
}

Polynomial::~Polynomial() {
    if (coefficients_ != nullptr) {
        cudaFree(coefficients_);
    }
}

__global__ void polyAddKernel(int n, int m, const Fr_t* a, const Fr_t* b, Fr_t* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i < m) {
            c[i] = blstrs__scalar__Scalar_add(a[i], b[i]);
        } else {
            c[i] = a[i];
        }
    } else if (i < m) {
        c[i] = b[i];
    }
}

Polynomial Polynomial::operator+(const Polynomial& other) {
    int resultDegree = max(degree_, other.degree_);
    Polynomial result(resultDegree);

    polyAddKernel<<<1, resultDegree + 1>>>(degree_ + 1, other.degree_ + 1, coefficients_, other.coefficients_, result.coefficients_);

    return result;
}

__global__ void polySubKernel(int n, int m, const Fr_t* a, const Fr_t* b, Fr_t* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i < m) {
            c[i] = blstrs__scalar__Scalar_sub(a[i], b[i]);
        } else {
            c[i] = a[i];
        }
    } else if (i < m) {
        c[i] = b[i];
    }
}

Polynomial Polynomial::operator-(const Polynomial& other) {
    int resultDegree = max(degree_, other.degree_);
    Polynomial result(resultDegree);

    polySubKernel<<<1, resultDegree + 1>>>(degree_ + 1, other.degree_ + 1, coefficients_, other.coefficients_, result.coefficients_);

    return result;
}

__global__ void polyMulKernel(int n, int m, const Fr_t* a, const Fr_t* b, Fr_t* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n + m - 1) {
        c[i] = blstrs__scalar__Scalar_ZERO;
        for (int j = max(0, i - m + 1); j <= min(i, n - 1); j++) {
            c[i] = blstrs__scalar__Scalar_add(c[i], blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a[j], b[i - j])));
        }
    }
}

Polynomial Polynomial::operator*(const Polynomial& other) {
    int resultDegree = degree_ + other.degree_;
    Polynomial result(resultDegree);

    polyMulKernel<<<1, resultDegree + 1>>>(degree_ + 1, other.degree_ + 1, coefficients_, other.coefficients_, result.coefficients_);

    return result;
}

Polynomial& Polynomial::operator=(const Polynomial& other) {
    if (coefficients_ != nullptr) {
        cudaFree(coefficients_);
    }
    degree_ = other.degree_;
    cudaMalloc((void**)&coefficients_, (degree_ + 1) * sizeof(Fr_t));
    cudaMemcpy(coefficients_, other.coefficients_, (degree_ + 1) * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    return *this;
}

__global__ void polyNegKernel(int n, const Fr_t* a, Fr_t* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ZERO, a[i]);
    }
}

Polynomial Polynomial::operator-() {
    Polynomial result(degree_);
    polyNegKernel<<<1, degree_ + 1>>>(degree_ + 1, coefficients_, result.coefficients_);
    return result;
}

// operator+=   
Polynomial& Polynomial::operator+=(const Polynomial& other)
{
    (*this) = (*this) + other;
    return *this;
}

// operator-=
Polynomial& Polynomial::operator-=(const Polynomial& other)
{
    (*this) = (*this) - other;
    return *this;
}

// operator*=
Polynomial& Polynomial::operator*=(const Polynomial& other)
{
    (*this) = (*this) * other;
    return *this;
}

__global__ void polyEvalKernel(int deg, const Fr_t* coefs, Fr_t x, Fr_t* result_ptr) {
    Fr_t pow = blstrs__scalar__Scalar_ONE;
    *result_ptr = blstrs__scalar__Scalar_ZERO;
    for (int i = 0; i <= deg; ++ i)
    {
        *result_ptr = blstrs__scalar__Scalar_add(*result_ptr, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(coefs[i], pow)));
        pow = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(pow, x));
    }
    *result_ptr = blstrs__scalar__Scalar_unmont(*result_ptr);
}

Fr_t Polynomial::operator()(const Fr_t& x)
{
    Fr_t* result_ptr;
    cudaMalloc((void**)&result_ptr, sizeof(Fr_t));
    polyEvalKernel<<<1, 1>>>(degree_, coefficients_, x, result_ptr);
    Fr_t result;
    cudaMemcpy(&result, result_ptr, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(result_ptr);
    return result;
}

int Polynomial::getDegree() const {
    return degree_;
}

void Polynomial::setCoefficients(int degree, Fr_t* coefficients){
    if (coefficients_ != nullptr) {
        cudaFree(coefficients_);
    }
    degree_ = degree;
    cudaMalloc((void**)&coefficients_, (degree_ + 1) * sizeof(Fr_t));
    cudaMemcpy(coefficients_, coefficients, (degree_ + 1) * sizeof(Fr_t), cudaMemcpyHostToDevice);
}

std::ostream& operator<<(std::ostream& os, const Polynomial& poly)
{
    Fr_t* coefficients = new Fr_t[poly.degree_ + 1];
    cudaMemcpy(coefficients, poly.coefficients_, (poly.degree_ + 1) * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i <= poly.degree_; i++) {
        os << coefficients[i] << " ";
    }
    delete[] coefficients;
    return os;
}

__global__ void eqPolyKernel(Fr_t u, Fr_t* coefs)
{
    coefs[0] = blstrs__scalar__Scalar_sub(FR_FROM_INT(1), u);
    coefs[1] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_double(u), FR_FROM_INT(1));
}

Polynomial Polynomial::eq(const Fr_t& u){
    Polynomial eq(1);
    eqPolyKernel<<<1, 1>>>(u, eq.coefficients_);
    return eq;
}

__global__ void eqEvalKernel(Fr_t u, Fr_t v, Fr_t* eval)
{
    *eval = blstrs__scalar__Scalar_double(blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(u, v)));
    *eval = blstrs__scalar__Scalar_sub(*eval, blstrs__scalar__Scalar_add(u, v));
    *eval = blstrs__scalar__Scalar_add(*eval, FR_FROM_INT(1));
}

Fr_t Polynomial::eq(const Fr_t& u, const Fr_t& v)
{
    Fr_t* eval;
    cudaMalloc((void**)&eval, sizeof(Fr_t));
    eqEvalKernel<<<1, 1>>>(u, v, eval);
    Fr_t result;
    cudaMemcpy(&result, eval, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(eval);
    return result;
}

#ifdef USE_GOLDILOCKS
// ── Lagrange interpolation from evaluation form ─────────────────────────────
// Given evals[k] = p(k) for k = 0,1,...,d, recover the degree-d polynomial
// in coefficient form.
//
// Uses the Lagrange basis: p(X) = sum_j evals[j] * L_j(X)
// where L_j(X) = prod_{k!=j} (X - k) / (j - k)
//
// We expand each L_j into coefficients and accumulate. This runs on the CPU
// since d is tiny (4-5 for our use case).

Polynomial Polynomial::from_evaluations(const std::vector<Fr_t>& evals) {
    if (evals.empty()) return Polynomial();
    uint d = evals.size() - 1;  // degree

    // We'll build the result in coefficient form on the CPU, then construct
    // the Polynomial (which copies to GPU).
    //
    // For each basis polynomial L_j(X) = prod_{k!=j} (X-k)/(j-k):
    // 1. Compute the scalar denominator 1/prod_{k!=j}(j-k)
    // 2. Expand prod_{k!=j}(X-k) as a polynomial in coefficient form
    // 3. Scale by evals[j] * (1/denominator) and accumulate

    std::vector<Fr_t> result_coeffs(d + 1, FR_ZERO);

    for (uint j = 0; j <= d; j++) {
        // Compute denominator: prod_{k!=j} (j - k)
        Fr_t denom = FR_ONE;
        for (uint k = 0; k <= d; k++) {
            if (k == j) continue;
            Fr_t diff;
            if (j > k) {
                diff = FR_FROM_INT(j - k);
            } else {
                diff = FR_ZERO - FR_FROM_INT(k - j);
            }
            denom = denom * diff;
        }
        Fr_t inv_denom = inv(denom);
        Fr_t scale = evals[j] * inv_denom;

        // Expand prod_{k!=j} (X - k) into coefficients
        // Start with polynomial = 1 (degree 0)
        std::vector<Fr_t> basis(1, FR_ONE);
        for (uint k = 0; k <= d; k++) {
            if (k == j) continue;
            // Multiply basis by (X - k)
            Fr_t neg_k;
            if (k == 0) {
                neg_k = FR_ZERO;
            } else {
                neg_k = FR_ZERO - FR_FROM_INT(k);
            }
            std::vector<Fr_t> new_basis(basis.size() + 1, FR_ZERO);
            for (uint m = 0; m < basis.size(); m++) {
                // basis[m] * X -> new_basis[m+1]
                new_basis[m + 1] = new_basis[m + 1] + basis[m];
                // basis[m] * (-k) -> new_basis[m]
                new_basis[m] = new_basis[m] + basis[m] * neg_k;
            }
            basis = new_basis;
        }

        // Accumulate: result += scale * basis
        for (uint m = 0; m < basis.size() && m <= d; m++) {
            result_coeffs[m] = result_coeffs[m] + scale * basis[m];
        }
    }

    return Polynomial(result_coeffs);
}
#endif // USE_GOLDILOCKS

// dummy