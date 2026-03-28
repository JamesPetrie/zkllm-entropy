# Compilers
NVCC := /usr/local/cuda/bin/nvcc

# Include and library paths
INCLUDES := -I$(CONDA_PREFIX)/include
LIBS := -L$(CONDA_PREFIX)/lib

# get compute capability from retrieved value
ARCH := sm_90

# NVCC compiler flags
# Note: -dlto (device link-time optimization) is omitted here to keep build times
# short (~5 min vs ~17 min). Add -dlto to the compile and link rules below for an
# optimized production build:
#   %.o: %.cu  →  add -dlto after -dc
#   %.o: %.cpp →  add -dlto after -dc
#   $(TARGETS) →  add -dlto at the end of the link command
NVCC_FLAGS := -arch=$(ARCH) -std=c++17 -O3

# Source and object files
CU_SRCS := bls12-381.cu ioutils.cu commitment.cu fr-tensor.cu g1-tensor.cu proof.cu zkrelu.cu zkfc.cu tlookup.cu polynomial.cu zksoftmax.cu rescaling.cu \
           zkargmax.cu zklog.cu zknormalcdf.cu zkentropy.cu
CU_OBJS := $(CU_SRCS:.cu=.o)
CPP_SRCS := $(wildcard *.cpp)
CPP_OBJS := $(CPP_SRCS:.cpp=.o)
TARGETS := main ppgen commit-param self-attn ffn rmsnorm skip-connection \
           zkllm_entropy commit_logits test_zkargmax test_zklog test_zknormalcdf test_zkentropy \
           zkllm_entropy_timed bench_field_arith bench_commitment
TARGET_OBJS := $(TARGETS:=.o)

# Pattern rule for CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# Pattern rule for C++ source files
%.o: %.cpp
	$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# General rule for building each target
$(TARGETS): % : %.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

# ── Standalone Goldilocks targets (no BLS12-381 dependency) ──────────────────
GOLD_FLAG := -DUSE_GOLDILOCKS
GOLD_SRCS := goldilocks.cu fr-tensor.cu proof.cu polynomial.cu ioutils.cu ntt.cu merkle.cu fri.cu fri_pcs.cu \
             zkrelu.cu zkfc.cu tlookup.cu zksoftmax.cu rescaling.cu zkargmax.cu zklog.cu zknormalcdf.cu zkentropy.cu
GOLD_OBJS := $(GOLD_SRCS:%.cu=gold_%.o)

gold_%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(GOLD_FLAG) $(INCLUDES) -dc $< -o $@

# Also handle .cpp files needed (timer.cpp)
gold_%.o: %.cpp
	$(NVCC) -x cu $(NVCC_FLAGS) $(GOLD_FLAG) $(INCLUDES) -dc $< -o $@

test_goldilocks: test_goldilocks.o goldilocks.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

test_gold_tensor: gold_test_gold_tensor.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

test_ntt: gold_test_ntt.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

test_merkle: gold_test_merkle.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

test_fri: gold_test_fri.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

test_fri_pcs: gold_test_fri_pcs.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_zkllm_entropy: gold_zkllm_entropy.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

bench_field: bench_field.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_bench_field: gold_bench_field.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_zkllm_entropy_timed: gold_zkllm_entropy_timed.o $(GOLD_OBJS) gold_timer.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_test_zkargmax: gold_test_zkargmax.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_test_zkentropy: gold_test_zkentropy.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_self-attn: gold_self-attn.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_ffn: gold_ffn.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_rmsnorm: gold_rmsnorm.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_skip-connection: gold_skip-connection.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_bench_matmul: gold_bench_matmul.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_commit-param: gold_commit-param.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# Clean rule
clean:
	rm -f $(TARGET_OBJS) $(CU_OBJS) $(CPP_OBJS) $(TARGETS) test_goldilocks test_goldilocks.o goldilocks.o $(GOLD_OBJS) gold_test_gold_tensor.o test_gold_tensor gold_test_ntt.o test_ntt gold_test_merkle.o test_merkle gold_test_fri.o test_fri gold_test_fri_pcs.o test_fri_pcs gold_fri_pcs.o

# ── CPU-only verifier targets (no CUDA dependency) ─────────────────────────
# These build with g++ only and can run on any machine without a GPU.

CXX := g++
CXX_FLAGS := -std=c++17 -O2 -DUSE_GOLDILOCKS

verifier: verifier.cpp verifier_utils.h sumcheck_verifier.h tlookup_verifier.h
	$(CXX) $(CXX_FLAGS) -o $@ verifier.cpp -lm

test_verifier: test_verifier.cpp verifier_utils.h sumcheck_verifier.h tlookup_verifier.h
	$(CXX) $(CXX_FLAGS) -o $@ test_verifier.cpp -lm

cpu: verifier test_verifier

# Default rule
all: $(TARGETS)

gold_rmsnorm_linear: gold_rmsnorm_linear.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_post_attn: gold_post_attn.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_layer_server: gold_layer_server.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
