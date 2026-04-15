# Compilers
NVCC := /usr/local/cuda/bin/nvcc

# Include and library paths.
#
# BLST is a host-side dependency used by src/field/hash_to_curve.* for
# RFC 9380 hash-to-curve (Phase 1.5, no toxic-waste generators).  On the
# H100 (jpetrieamodo) the library is built from source at ~/blst/; set
# BLST_DIR to override for other installs.
BLST_DIR ?= $(HOME)/blst
INCLUDES := -I$(CONDA_PREFIX)/include -Isrc -I$(BLST_DIR)/bindings
LIBS := -L$(CONDA_PREFIX)/lib -L$(BLST_DIR)
EXTRA_LIBS := -lblst

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

# ── BLS12-381 source files ──────────────────────────────────────────────────
CU_SRCS := src/field/bls12-381.cu src/field/hash_to_curve.cu \
           src/util/ioutils.cu src/commit/commitment.cu \
           src/tensor/fr-tensor.cu src/tensor/g1-tensor.cu src/proof/proof.cu \
           src/proof/hyrax_sigma.cu \
           src/zknn/zkrelu.cu src/zknn/zkfc.cu src/zknn/tlookup.cu \
           src/poly/polynomial.cu src/zknn/zksoftmax.cu src/zknn/rescaling.cu \
           src/zknn/zkargmax.cu src/zknn/zklog.cu src/zknn/zknormalcdf.cu \
           src/entropy/zkentropy.cu
CU_OBJS := $(patsubst src/%.cu,$(BUILD)/%.o,$(CU_SRCS))

CPP_SRCS := src/util/timer.cpp
CPP_OBJS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(CPP_SRCS))

# Entry-point targets
BLS_TARGETS := main ppgen commit-param self-attn ffn rmsnorm skip-connection \
               zkllm_entropy commit_logits test_zkargmax test_zklog test_zknormalcdf test_zkentropy \
               test_hiding_pedersen test_open_zk test_verify_weight_zk \
               test_opening_distinguisher \
               test_hash_to_curve_rfc9380 test_htc_generators test_pp_format \
               test_hyrax_sigma \
               zkllm_entropy_timed bench_field_arith bench_commitment

# Create build subdirectories
$(shell mkdir -p $(BUILD)/field $(BUILD)/poly $(BUILD)/commit $(BUILD)/proof \
                 $(BUILD)/tensor $(BUILD)/zknn $(BUILD)/entropy $(BUILD)/llm $(BUILD)/util \
                 $(BUILD)/bin $(BUILD)/test $(BUILD)/bench)

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

# ── Link rules ──────────────────────────────────────────────────────────────

# Entry points in bin/
main: $(BUILD)/bin/main.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

ppgen: $(BUILD)/bin/ppgen.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

commit-param: $(BUILD)/bin/commit-param.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

commit_logits: $(BUILD)/bin/commit_logits.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

zkllm_entropy: $(BUILD)/bin/zkllm_entropy.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

zkllm_entropy_timed: $(BUILD)/bin/zkllm_entropy_timed.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

# LLM layer targets (source in src/llm/)
self-attn: $(BUILD)/llm/self-attn.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

ffn: $(BUILD)/llm/ffn.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

rmsnorm: $(BUILD)/llm/rmsnorm.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

skip-connection: $(BUILD)/llm/skip-connection.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

# Test targets (source in test/)
test_zkargmax: $(BUILD)/test/test_zkargmax.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

test_zklog: $(BUILD)/test/test_zklog.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

test_zknormalcdf: $(BUILD)/test/test_zknormalcdf.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

test_zkentropy: $(BUILD)/test/test_zkentropy.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

test_hiding_pedersen: $(BUILD)/test/test_hiding_pedersen.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

test_open_zk: $(BUILD)/test/test_open_zk.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

test_verify_weight_zk: $(BUILD)/test/test_verify_weight_zk.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

test_opening_distinguisher: $(BUILD)/test/test_opening_distinguisher.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

test_hash_to_curve_rfc9380: $(BUILD)/test/test_hash_to_curve_rfc9380.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

test_htc_generators: $(BUILD)/test/test_htc_generators.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

test_pp_format: $(BUILD)/test/test_pp_format.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

test_hyrax_sigma: $(BUILD)/test/test_hyrax_sigma.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

# Bench targets (source in bench/)
bench_field_arith: $(BUILD)/bench/bench_field_arith.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

bench_commitment: $(BUILD)/bench/bench_commitment.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ $(EXTRA_LIBS) -o $@

# ── Clean rule ───────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD)
	rm -f $(BLS_TARGETS)

# Default rule
all: $(BLS_TARGETS)

.PHONY: all clean
