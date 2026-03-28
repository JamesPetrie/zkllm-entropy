# Task: Generate Hardware Nondeterminism Calibration Data for Llama-2-7B

## Goal
Run multiple independent inference passes on the same prompts to measure natural hardware nondeterminism on H100. This data will calibrate the σ parameter for a zero-knowledge conditional entropy proof system.

## Environment Setup

1. Confirm you have access to an H100 GPU:
```bash
nvidia-smi
```

2. Create and activate a conda environment:
```bash
conda create -n nondeterminism python=3.10 -y
conda activate nondeterminism
pip install torch transformers accelerate safetensors
pip install flash-attn --no-build-isolation
```

The `flash-attn` install compiles from source and may take several minutes. If it fails, note the error — the script will fall back to SDPA, but Flash Attention is the realistic frontier setting we want.

3. Log in to HuggingFace (Llama-2-7B requires accepting Meta's license at https://huggingface.co/meta-llama/Llama-2-7b-hf):
```bash
huggingface-cli login
```

4. Verify CUDA and Flash Attention:
```python
python -c "import torch; print(torch.cuda.get_device_name())"
python -c "from flash_attn import flash_attn_func; print('Flash Attention available')"
```

## Implementation

Create a file called `collect_nondeterminism.py`:

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from datetime import datetime
from collections import Counter
from scipy.optimize import minimize_scalar
from scipy.stats import norm

TOP_K = 1000  # Store top-1000 logits per position

TEST_PROMPTS = [
    # Confident completions (expect high match rates)
    "The capital of France is",
    "The three laws of thermodynamics are: 1.",
    "The solution to the equation x^2 - 5x + 6 = 0 is",
    # Code (structured, some ambiguity in style choices)
    "def fibonacci(n):\n    '''Return the nth Fibonacci number.'''\n",
    "import numpy as np\ndef solve_linear_system(A, b):\n    '''Solve Ax = b.'''\n",
    # Open-ended / creative (more close-call positions expected)
    "In the year 2024, artificial intelligence",
    "Once upon a time in a land far away,",
    "According to quantum mechanics, particles can",
    "The best approach to solving climate change involves",
    "There are several reasons why the Roman Empire fell, including",
    # Enumeration / list-style (multiple plausible orderings)
    "The top five programming languages in 2024 are:",
    "Here is a list of common machine learning algorithms:",
]


def load_model():
    """Load model with realistic frontier inference settings."""
    model_name = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Try Flash Attention 2, fall back to SDPA
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        print("Using Flash Attention 2")
    except Exception as e:
        print(f"Flash Attention 2 unavailable ({e}), falling back to SDPA")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
    model.eval()

    return model, tokenizer


def single_inference_pass(model, tokenizer, prompt, max_new_tokens=64):
    """Run one inference pass with KV caching, returning tokens and top-k logits."""

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

    generated_tokens = outputs.sequences[0, input_ids.shape[1]:].tolist()

    # Extract top-k logits per position (much smaller than full 32K vocab)
    top_k_data = []
    for logits_tensor in outputs.logits:
        logits = logits_tensor[0].float()  # [vocab_size], convert to float32
        top_values, top_indices = torch.topk(logits, TOP_K)
        top_k_data.append({
            "values": top_values.cpu().numpy(),
            "indices": top_indices.cpu().numpy(),
        })

    return {
        "tokens": generated_tokens,
        "top_k": top_k_data,
    }


def collect_samples(model, tokenizer, prompt, n_runs=20, max_new_tokens=64,
                    output_dir="./nondeterminism_data"):
    """Collect multiple independent runs for one prompt."""

    os.makedirs(output_dir, exist_ok=True)
    prompt_hash = hex(abs(hash(prompt)) % (10**8))[2:]

    all_runs = []
    for run_idx in range(n_runs):
        print(f"  Run {run_idx + 1}/{n_runs}...", end=" ", flush=True)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        result = single_inference_pass(model, tokenizer, prompt, max_new_tokens)
        result["run_idx"] = run_idx
        result["timestamp"] = datetime.now().isoformat()

        all_runs.append(result)
        print(f"generated {len(result['tokens'])} tokens")

    # Save as numpy binary for compact storage
    save_path = os.path.join(output_dir, f"prompt_{prompt_hash}")
    os.makedirs(save_path, exist_ok=True)

    # Save metadata as JSON (small)
    meta = {
        "prompt": prompt,
        "n_runs": n_runs,
        "max_new_tokens": max_new_tokens,
        "top_k": TOP_K,
        "runs": [],
    }

    for run in all_runs:
        n_positions = len(run["tokens"])

        # Save logit data as numpy arrays
        values = np.stack([pos["values"] for pos in run["top_k"]])  # [n_pos, TOP_K]
        indices = np.stack([pos["indices"] for pos in run["top_k"]])  # [n_pos, TOP_K]

        run_id = run["run_idx"]
        np.save(os.path.join(save_path, f"run_{run_id}_values.npy"), values)
        np.save(os.path.join(save_path, f"run_{run_id}_indices.npy"), indices)

        meta["runs"].append({
            "run_idx": run_id,
            "tokens": run["tokens"],
            "timestamp": run["timestamp"],
            "n_positions": n_positions,
        })

    with open(os.path.join(save_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {save_path}/")
    return all_runs, save_path


def analyze_matches(all_runs):
    """Analyze token match rates, separating true nondeterminism from cascade.

    A 'true' disagreement is at a position where the prefix tokens were identical
    across both runs. A 'cascade' disagreement is where a prior token already differed,
    so the inputs to the model were different.
    """
    n_runs = len(all_runs)
    tokens_per_run = [r["tokens"] for r in all_runs]
    min_len = min(len(t) for t in tokens_per_run)

    # Pairwise analysis
    true_disagree = 0       # Disagreement with identical prefix
    cascade_disagree = 0    # Disagreement after prior mismatch
    agree = 0
    first_divergence_positions = []

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            found_first_divergence = False
            for pos in range(min_len):
                if tokens_per_run[i][pos] == tokens_per_run[j][pos]:
                    if not found_first_divergence:
                        agree += 1
                    else:
                        # Tokens happen to match again after divergence
                        # (possible but the prefix was different, so still cascade)
                        cascade_disagree += 0  # Don't count agreements in cascade zone
                        agree += 1  # Actually they do agree, count it
                else:
                    if not found_first_divergence:
                        true_disagree += 1
                        first_divergence_positions.append(pos)
                        found_first_divergence = True
                    else:
                        cascade_disagree += 1

    total_comparable = agree + true_disagree + cascade_disagree
    n_pairs = n_runs * (n_runs - 1) // 2

    print(f"\nMatch rate analysis ({n_pairs} run pairs, {min_len} positions each):")
    print(f"  Token agreements:          {agree}/{total_comparable} ({agree/total_comparable:.4f})")
    print(f"  True disagreements:        {true_disagree} (hardware nondeterminism)")
    print(f"  Cascade disagreements:     {cascade_disagree} (follow-on from prior mismatch)")
    print(f"  True disagreement rate:    {true_disagree/total_comparable:.6f}")

    if first_divergence_positions:
        print(f"  First-divergence positions: {sorted(set(first_divergence_positions))[:15]}"
              f"{'...' if len(set(first_divergence_positions)) > 15 else ''}")

    # Per-position consensus (what fraction of runs agree with the mode)
    position_consensus = []
    for pos in range(min_len):
        tokens_at_pos = [t[pos] for t in tokens_per_run]
        counts = Counter(tokens_at_pos)
        mode_count = counts.most_common(1)[0][1]
        position_consensus.append(mode_count / n_runs)

    weak_positions = [(i, c) for i, c in enumerate(position_consensus) if c < 1.0]
    if weak_positions:
        print(f"  Positions with <100% consensus ({len(weak_positions)}):")
        for pos, cons in weak_positions[:10]:
            tokens_at_pos = [t[pos] for t in tokens_per_run]
            counts = Counter(tokens_at_pos)
            print(f"    pos {pos}: consensus {cons:.0%}, tokens: {dict(counts)}")

    return {
        "agree": agree,
        "true_disagree": true_disagree,
        "cascade_disagree": cascade_disagree,
        "total": total_comparable,
        "true_disagree_rate": true_disagree / total_comparable if total_comparable > 0 else 0,
        "first_divergence_positions": first_divergence_positions,
        "position_consensus": position_consensus,
    }


def analyze_logit_perturbations(all_runs, save_path):
    """Compare logit values across runs at pre-divergence positions.

    For each pair of runs, at each position before the first token divergence,
    compute the difference in logit values. This gives the empirical distribution
    of hardware noise on logits.
    """
    n_runs = len(all_runs)
    tokens_per_run = [r["tokens"] for r in all_runs]

    all_diffs = []        # All logit diffs at shared top-k indices
    top1_gaps = []        # Gap between top-1 and top-2 at disagreement positions
    top1_gaps_agree = []  # Gap between top-1 and top-2 at agreement positions

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            # Load logit data for this pair
            values_i = np.load(os.path.join(save_path, f"run_{i}_values.npy"))
            values_j = np.load(os.path.join(save_path, f"run_{j}_values.npy"))
            indices_i = np.load(os.path.join(save_path, f"run_{i}_indices.npy"))
            indices_j = np.load(os.path.join(save_path, f"run_{j}_indices.npy"))

            min_len = min(len(tokens_per_run[i]), len(tokens_per_run[j]))

            for pos in range(min_len):
                # Find tokens that appear in both top-k lists
                set_i = set(indices_i[pos].tolist())
                set_j = set(indices_j[pos].tolist())
                shared = set_i & set_j

                if not shared:
                    continue

                # Build lookup: token_id -> logit_value
                lookup_i = dict(zip(indices_i[pos].tolist(), values_i[pos].tolist()))
                lookup_j = dict(zip(indices_j[pos].tolist(), values_j[pos].tolist()))

                diffs = [lookup_i[t] - lookup_j[t] for t in shared]
                all_diffs.extend(diffs)

                # Track top-1 to top-2 gap
                gap_i = values_i[pos][0] - values_i[pos][1]
                gap_j = values_j[pos][0] - values_j[pos][1]
                avg_gap = (gap_i + gap_j) / 2

                if tokens_per_run[i][pos] != tokens_per_run[j][pos]:
                    top1_gaps.append(avg_gap)
                    break  # Stop at first divergence (cascade from here)
                else:
                    top1_gaps_agree.append(avg_gap)

    all_diffs = np.array(all_diffs)

    print(f"\nLogit perturbation analysis:")
    print(f"  Total logit diffs measured: {len(all_diffs)}")
    if len(all_diffs) > 0:
        print(f"  Mean diff:    {np.mean(all_diffs):.6f}")
        print(f"  Std diff:     {np.std(all_diffs):.6f}")
        print(f"  Max |diff|:   {np.max(np.abs(all_diffs)):.6f}")
        print(f"  Median |diff|: {np.median(np.abs(all_diffs)):.6f}")
        # Percentiles
        for p in [90, 95, 99, 99.9]:
            print(f"  |diff| p{p}: {np.percentile(np.abs(all_diffs), p):.6f}")

    if top1_gaps:
        print(f"\n  Top-1 to top-2 gap at DISAGREEMENT positions ({len(top1_gaps)} samples):")
        print(f"    Mean gap:   {np.mean(top1_gaps):.4f}")
        print(f"    Median gap: {np.median(top1_gaps):.4f}")
        print(f"    Min gap:    {np.min(top1_gaps):.4f}")

    if top1_gaps_agree:
        print(f"\n  Top-1 to top-2 gap at AGREEMENT positions ({len(top1_gaps_agree)} samples):")
        print(f"    Mean gap:   {np.mean(top1_gaps_agree):.4f}")
        print(f"    Median gap: {np.median(top1_gaps_agree):.4f}")

    return {
        "logit_diffs_std": float(np.std(all_diffs)) if len(all_diffs) > 0 else None,
        "logit_diffs_mean": float(np.mean(all_diffs)) if len(all_diffs) > 0 else None,
        "n_diffs": len(all_diffs),
        "top1_gaps_at_disagree": top1_gaps,
        "top1_gaps_at_agree": top1_gaps_agree,
    }


def calibrate_sigma(match_analysis, logit_analysis):
    """Estimate σ from the empirical data.

    Two approaches:
    1. From logit diffs directly: σ ≈ std(logit_diffs) / sqrt(2)
       (since diff of two samples each with noise σ has std σ√2)
    2. From match rate: find σ such that the noise model predicts
       the observed true disagreement rate given the observed logit gaps.
    """
    print(f"\n{'='*60}")
    print("σ CALIBRATION")
    print('='*60)

    # Method 1: Direct from logit perturbation std
    if logit_analysis["logit_diffs_std"] is not None:
        # Each diff is (logit + noise1) - (logit + noise2) = noise1 - noise2
        # If noise ~ N(0, σ²), then diff ~ N(0, 2σ²), so std(diff) = σ√2
        sigma_from_diffs = logit_analysis["logit_diffs_std"] / np.sqrt(2)
        print(f"\nMethod 1 (logit diff std):")
        print(f"  Observed std of logit diffs: {logit_analysis['logit_diffs_std']:.6f}")
        print(f"  Estimated σ = std/√2 = {sigma_from_diffs:.6f}")
        sigma_eff = int(sigma_from_diffs * 65536)
        print(f"  σ_eff (×65536) = {sigma_eff}")

    # Method 2: From match rate + logit gaps
    true_disagree_rate = match_analysis["true_disagree_rate"]
    gaps_agree = logit_analysis.get("top1_gaps_at_agree", [])
    gaps_disagree = logit_analysis.get("top1_gaps_at_disagree", [])

    if true_disagree_rate > 0 and len(gaps_agree) > 0:
        all_gaps = gaps_agree + gaps_disagree

        def predicted_flip_rate(sigma):
            """Probability that Gaussian noise flips the argmax, given observed gaps."""
            if sigma <= 0:
                return 0.0
            # P(flip) ≈ P(noise > gap/2) for each position
            # This is approximate; assumes only top-2 tokens compete
            rates = [2 * norm.sf(g / (sigma * np.sqrt(2))) for g in all_gaps]
            return np.mean(rates)

        def objective(log_sigma):
            sigma = np.exp(log_sigma)
            predicted = predicted_flip_rate(sigma)
            if predicted <= 0:
                return 1e10
            return (np.log(predicted) - np.log(true_disagree_rate))**2

        result = minimize_scalar(objective, bounds=(-10, 2), method='bounded')
        sigma_from_match = np.exp(result.x)
        predicted_rate = predicted_flip_rate(sigma_from_match)

        print(f"\nMethod 2 (match rate fitting):")
        print(f"  Observed true disagree rate: {true_disagree_rate:.6f}")
        print(f"  Estimated σ = {sigma_from_match:.6f}")
        print(f"  Predicted disagree rate at this σ: {predicted_rate:.6f}")
        sigma_eff_2 = int(sigma_from_match * 65536)
        print(f"  σ_eff (×65536) = {sigma_eff_2}")
    else:
        print(f"\nMethod 2: Skipped (no disagreements observed or no gap data)")
        sigma_from_match = None

    return {
        "sigma_from_diffs": float(sigma_from_diffs) if logit_analysis["logit_diffs_std"] else None,
        "sigma_from_match": float(sigma_from_match) if sigma_from_match else None,
    }


def main():
    print("Loading model...")
    model, tokenizer = load_model()

    print(f"\nModel loaded. Device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Deterministic algorithms enabled: {torch.are_deterministic_algorithms_enabled()}")

    all_match_analyses = []
    all_logit_analyses = []
    all_save_paths = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'='*60}")
        print(f"Prompt {i+1}/{len(TEST_PROMPTS)}: {prompt[:50]}...")
        print('='*60)

        all_runs, save_path = collect_samples(
            model, tokenizer, prompt,
            n_runs=20,
            max_new_tokens=64,
            output_dir="./nondeterminism_data"
        )
        all_save_paths.append(save_path)

        match_analysis = analyze_matches(all_runs)
        logit_analysis = analyze_logit_perturbations(all_runs, save_path)

        all_match_analyses.append(match_analysis)
        all_logit_analyses.append(logit_analysis)

    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY")
    print('='*60)

    total_agree = sum(m["agree"] for m in all_match_analyses)
    total_true = sum(m["true_disagree"] for m in all_match_analyses)
    total_cascade = sum(m["cascade_disagree"] for m in all_match_analyses)
    total_all = total_agree + total_true + total_cascade

    print(f"Total token comparisons:     {total_all}")
    print(f"Total agreements:            {total_agree} ({total_agree/total_all:.4f})")
    print(f"Total true disagreements:    {total_true} ({total_true/total_all:.6f})")
    print(f"Total cascade disagreements: {total_cascade}")

    # Aggregate logit analysis
    all_stds = [la["logit_diffs_std"] for la in all_logit_analyses if la["logit_diffs_std"] is not None]
    if all_stds:
        print(f"\nLogit diff std across prompts: {np.mean(all_stds):.6f} (mean), "
              f"{np.std(all_stds):.6f} (variation)")

    # Calibrate σ from aggregate data
    aggregate_match = {
        "true_disagree_rate": total_true / total_all if total_all > 0 else 0,
    }
    aggregate_logit = {
        "logit_diffs_std": np.mean(all_stds) if all_stds else None,
        "logit_diffs_mean": None,
        "n_diffs": sum(la["n_diffs"] for la in all_logit_analyses),
        "top1_gaps_at_agree": [g for la in all_logit_analyses for g in la.get("top1_gaps_at_agree", [])],
        "top1_gaps_at_disagree": [g for la in all_logit_analyses for g in la.get("top1_gaps_at_disagree", [])],
    }

    sigma_results = calibrate_sigma(aggregate_match, aggregate_logit)

    # Save final summary
    summary = {
        "total_agree": total_agree,
        "total_true_disagree": total_true,
        "total_cascade_disagree": total_cascade,
        "overall_true_disagree_rate": total_true / total_all if total_all > 0 else 0,
        "sigma_estimates": sigma_results,
        "per_prompt": [
            {
                "prompt": TEST_PROMPTS[i],
                "agree": all_match_analyses[i]["agree"],
                "true_disagree": all_match_analyses[i]["true_disagree"],
                "cascade_disagree": all_match_analyses[i]["cascade_disagree"],
                "logit_diffs_std": all_logit_analyses[i]["logit_diffs_std"],
            }
            for i in range(len(TEST_PROMPTS))
        ],
    }

    with open("./nondeterminism_data/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFull summary saved to ./nondeterminism_data/summary.json")


if __name__ == "__main__":
    main()
```

## Running the Collection

```bash
conda activate nondeterminism
python collect_nondeterminism.py
```

Expected runtime: ~3-5 minutes on H100.

## Output

The script creates `./nondeterminism_data/` containing:
- One directory per prompt with numpy arrays for top-1000 logit values/indices per run
- `meta.json` per prompt with token sequences and run metadata
- `summary.json` with aggregate statistics and σ estimates

## What to Report Back

1. The full console output (copy-paste everything)
2. The contents of `./nondeterminism_data/summary.json`
3. Whether Flash Attention 2 loaded or fell back to SDPA
4. If true disagreement rate is 0% across all prompts, report the output of:
   ```python
   python -c "import torch; print(torch.are_deterministic_algorithms_enabled())"
   ```
   and we may need more runs or longer sequences.

## Notes

- If `flash-attn` fails to install, proceed without it but note this in results. SDPA is also a realistic backend but has different nondeterminism characteristics.
- The top-1000 logit storage uses ~4MB per prompt (20 runs × 64 positions × 1000 × 4 bytes × 2 arrays), compared to ~130GB for full vocab JSON. Total dataset is ~50MB.
- The σ calibration uses two independent methods: (1) direct logit perturbation std, (2) fitting to observed token flip rate. If they agree, the estimate is robust. If they diverge, the logit diff method is more reliable since it uses more data.
- `scipy` is required for the calibration step. Add `pip install scipy` if not already installed.
