# Compilers
NVCC := /usr/local/cuda/bin/nvcc

# Include and library paths
INCLUDES := -I$(CONDA_PREFIX)/include -Isrc
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

# Build directory for object files
BUILD := build

# ── BLS12-381 (original) source files ────────────────────────────────────────
CU_SRCS := src/field/bls12-381.cu src/util/ioutils.cu src/commit/commitment.cu \
           src/tensor/fr-tensor.cu src/tensor/g1-tensor.cu src/proof/proof.cu \
           src/zknn/zkrelu.cu src/zknn/zkfc.cu src/zknn/tlookup.cu \
           src/poly/polynomial.cu src/zknn/zksoftmax.cu src/zknn/rescaling.cu \
           src/zknn/zkargmax.cu src/zknn/zklog.cu src/zknn/zknormalcdf.cu \
           src/entropy/zkentropy.cu
CU_OBJS := $(patsubst src/%.cu,$(BUILD)/%.o,$(CU_SRCS))

CPP_SRCS := src/util/timer.cpp src/llm/skip_connection_cpu.cpp
CPP_OBJS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(CPP_SRCS))

# BLS12-381 entry-point targets
BLS_TARGETS := main ppgen commit-param self-attn ffn rmsnorm skip-connection \
               zkllm_entropy commit_logits test_zkargmax test_zklog test_zknormalcdf test_zkentropy \
               zkllm_entropy_timed bench_field_arith bench_commitment

# Create build subdirectories
$(shell mkdir -p $(BUILD)/field $(BUILD)/poly $(BUILD)/commit $(BUILD)/proof \
                 $(BUILD)/tensor $(BUILD)/zknn $(BUILD)/entropy $(BUILD)/llm $(BUILD)/util \
                 $(BUILD)/bin $(BUILD)/test $(BUILD)/bench \
                 $(BUILD)/gold/field $(BUILD)/gold/poly $(BUILD)/gold/commit $(BUILD)/gold/proof \
                 $(BUILD)/gold/tensor $(BUILD)/gold/zknn $(BUILD)/gold/entropy $(BUILD)/gold/llm $(BUILD)/gold/util \
                 $(BUILD)/gold/bin $(BUILD)/gold/test $(BUILD)/gold/bench)

# ── Pattern rules ────────────────────────────────────────────────────────────

# Library .cu files under src/
$(BUILD)/%.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# Library .cpp files under src/
$(BUILD)/%.o: src/%.cpp
	$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# Entry-point .cu files under bin/
$(BUILD)/bin/%.o: bin/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# Test .cu files under test/
$(BUILD)/test/%.o: test/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# Bench .cu files under bench/
$(BUILD)/bench/%.o: bench/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# ── BLS12-381 link rules ────────────────────────────────────────────────────

# Entry points in bin/
main: $(BUILD)/bin/main.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

ppgen: $(BUILD)/bin/ppgen.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

commit-param: $(BUILD)/bin/commit-param.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

commit_logits: $(BUILD)/bin/commit_logits.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

zkllm_entropy: $(BUILD)/bin/zkllm_entropy.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

zkllm_entropy_timed: $(BUILD)/bin/zkllm_entropy_timed.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

# LLM layer targets (source in src/llm/)
self-attn: $(BUILD)/llm/self-attn.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

ffn: $(BUILD)/llm/ffn.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

rmsnorm: $(BUILD)/llm/rmsnorm.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

skip-connection: $(BUILD)/llm/skip-connection.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

# Test targets (source in test/)
test_zkargmax: $(BUILD)/test/test_zkargmax.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

test_zklog: $(BUILD)/test/test_zklog.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

test_zknormalcdf: $(BUILD)/test/test_zknormalcdf.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

test_zkentropy: $(BUILD)/test/test_zkentropy.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

# Bench targets (source in bench/)
bench_field_arith: $(BUILD)/bench/bench_field_arith.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

bench_commitment: $(BUILD)/bench/bench_commitment.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

# ── Standalone Goldilocks targets (no BLS12-381 dependency) ──────────────────
GOLD_FLAG := -DUSE_GOLDILOCKS
GOLD_SRCS := src/field/goldilocks.cu src/tensor/fr-tensor.cu src/proof/proof.cu \
             src/poly/polynomial.cu src/util/ioutils.cu src/poly/ntt.cu \
             src/commit/merkle.cu src/commit/fri.cu src/commit/fri_pcs.cu \
             src/zknn/zkrelu.cu src/zknn/zkfc.cu src/zknn/tlookup.cu \
             src/zknn/zksoftmax.cu src/zknn/rescaling.cu src/zknn/zkargmax.cu \
             src/zknn/zklog.cu src/zknn/zknormalcdf.cu src/entropy/zkentropy.cu
GOLD_OBJS := $(patsubst src/%.cu,$(BUILD)/gold/%.o,$(GOLD_SRCS))

# Goldilocks pattern rules
$(BUILD)/gold/%.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) $(GOLD_FLAG) $(INCLUDES) -dc $< -o $@

$(BUILD)/gold/%.o: src/%.cpp
	$(NVCC) -x cu $(NVCC_FLAGS) $(GOLD_FLAG) $(INCLUDES) -dc $< -o $@

$(BUILD)/gold/bin/%.o: bin/%.cu
	$(NVCC) $(NVCC_FLAGS) $(GOLD_FLAG) $(INCLUDES) -dc $< -o $@

$(BUILD)/gold/test/%.o: test/%.cu
	$(NVCC) $(NVCC_FLAGS) $(GOLD_FLAG) $(INCLUDES) -dc $< -o $@

$(BUILD)/gold/bench/%.o: bench/%.cu
	$(NVCC) $(NVCC_FLAGS) $(GOLD_FLAG) $(INCLUDES) -dc $< -o $@

$(BUILD)/gold/llm/%.o: src/llm/%.cu
	$(NVCC) $(NVCC_FLAGS) $(GOLD_FLAG) $(INCLUDES) -dc $< -o $@

# ── Goldilocks link targets ─────────────────────────────────────────────────

test_goldilocks: $(BUILD)/test/test_goldilocks.o $(BUILD)/field/goldilocks.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

test_gold_tensor: $(BUILD)/gold/test/test_gold_tensor.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

test_ntt: $(BUILD)/gold/test/test_ntt.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

test_merkle: $(BUILD)/gold/test/test_merkle.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

test_fri: $(BUILD)/gold/test/test_fri.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

test_fri_pcs: $(BUILD)/gold/test/test_fri_pcs.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_zkllm_entropy: $(BUILD)/gold/bin/zkllm_entropy.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_zkllm_entropy_timed: $(BUILD)/gold/bin/zkllm_entropy_timed.o $(GOLD_OBJS) $(BUILD)/gold/util/timer.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_test_zkargmax: $(BUILD)/gold/test/test_zkargmax.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_test_zkentropy: $(BUILD)/gold/test/test_zkentropy.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_self-attn: $(BUILD)/gold/llm/self-attn.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_ffn: $(BUILD)/gold/llm/ffn.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_rmsnorm: $(BUILD)/gold/llm/rmsnorm.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_skip-connection: $(BUILD)/gold/llm/skip-connection.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_commit-param: $(BUILD)/gold/bin/commit-param.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

bench_field: $(BUILD)/bench/bench_field.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_bench_field: $(BUILD)/gold/bench/bench_field.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_bench_matmul: $(BUILD)/gold/bench/bench_matmul.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_rmsnorm_linear: $(BUILD)/gold/llm/rmsnorm_linear.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_post_attn: $(BUILD)/gold/llm/post_attn.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

gold_layer_server: $(BUILD)/gold/bin/layer_server.o $(GOLD_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# ── CPU-only verifier targets (no CUDA dependency) ─────────────────────────
# These build with g++ only and can run on any machine without a GPU.

CXX := g++
CXX_FLAGS := -std=c++17 -O2 -DUSE_GOLDILOCKS

entropy_verifier: verifier/verifier.cpp verifier/verifier_utils.h verifier/sumcheck_verifier.h verifier/tlookup_verifier.h
	$(CXX) $(CXX_FLAGS) -I verifier -o $@ verifier/verifier.cpp -lm

test_verifier: test/test_verifier.cpp verifier/verifier_utils.h verifier/sumcheck_verifier.h verifier/tlookup_verifier.h
	$(CXX) $(CXX_FLAGS) -I verifier -o $@ test/test_verifier.cpp -lm

cpu: entropy_verifier test_verifier

# ── Clean rule ───────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD)
	rm -f $(BLS_TARGETS) test_goldilocks test_gold_tensor test_ntt test_merkle \
	      test_fri test_fri_pcs gold_zkllm_entropy gold_zkllm_entropy_timed \
	      gold_test_zkargmax gold_test_zkentropy gold_self-attn gold_ffn gold_rmsnorm \
	      gold_skip-connection gold_commit-param bench_field gold_bench_field \
	      gold_bench_matmul gold_rmsnorm_linear gold_post_attn gold_layer_server \
	      entropy_verifier test_verifier

# Default rule
all: $(BLS_TARGETS)

.PHONY: all clean cpu
