# References and Related Work

A comprehensive list of papers, code repositories, textbooks, and blog posts relevant to zkllm-entropy.

---

## Table of Contents

1. [zkML and zkLLM — Core Papers](#1-zkml-and-zkllm--core-papers)
2. [zkML Frameworks and Code Repositories](#2-zkml-frameworks-and-code-repositories)
3. [Steganography, Covert Channels, and Information Hiding in LLMs](#3-steganography-covert-channels-and-information-hiding-in-llms)
4. [Model Weight Exfiltration and Inference Verification](#4-model-weight-exfiltration-and-inference-verification)
5. [GPU Floating-Point Non-Determinism](#5-gpu-floating-point-non-determinism)
6. [AI Governance, Compute Verification, and International Agreements](#6-ai-governance-compute-verification-and-international-agreements)
7. [Hardware Attestation and Trusted Execution Environments](#7-hardware-attestation-and-trusted-execution-environments)
8. [LLM Watermarking and Entropy](#8-llm-watermarking-and-entropy)
9. [Zero-Knowledge Proof Primitives and Cryptography](#9-zero-knowledge-proof-primitives-and-cryptography)
10. [Textbooks and Surveys](#10-textbooks-and-surveys)
11. [Blog Posts and Educational Resources](#11-blog-posts-and-educational-resources)

---

## 1. zkML and zkLLM — Core Papers

**zkLLM: Zero Knowledge Proofs for Large Language Models**
Sun, Li, Zhang. ACM CCS 2024.
The first specialized ZKP system for LLMs. Introduces tlookup (parallelized lookup argument for non-arithmetic tensor operations) and zkAttn (ZKP for attention mechanisms). Proves inference of a 13B-parameter model in under 15 minutes.
[arXiv](https://arxiv.org/abs/2404.16109) | [GitHub](https://github.com/jvhs0706/zkllm-ccs2024)

**zkGPT: An Efficient Non-interactive Zero-knowledge Proof Framework for LLM Inference**
Qu, Sun, Liu, Lu, Guo, Chen, Zhang. USENIX Security 2025.
First practical ZKP for LLMs under general settings. Proves GPT-2 inference in under 25 seconds. Introduces constraint fusion for non-linear layers and circuit squeeze for parallelism.
[ePrint 2025/1184](https://eprint.iacr.org/2025/1184) | [USENIX](https://www.usenix.org/conference/usenixsecurity25/presentation/qu-zkgpt)

**ZKTorch: Compiling ML Inference to Zero-Knowledge Proofs via Parallel Proof Accumulation**
Kang et al. 2025.
First universal ZKML compiler supporting every edge model in MLPerf Inference mobile suite. Proves a 6B-parameter LLM (GPT-J) in ~20 minutes on 64 threads.
[arXiv](https://arxiv.org/abs/2507.07031) | [GitHub](https://github.com/uiuc-kang-lab/zk-torch) | [Blog](https://ddkang.substack.com/p/zktorch-open-sourcing-the-first-universal)

**zkPyTorch: A Hierarchical Optimized Compiler for Zero-Knowledge Machine Learning**
Polyhedra Network. 2025.
Transforms standard PyTorch/ONNX models into ZKP-compatible circuits. Achieves Llama-3 inference proof in 150 seconds per token.
[ePrint 2025/535](https://eprint.iacr.org/2025/535) | [Blog](https://blog.polyhedra.network/zkpytorch/)

**Folding-based zkLLM**
Wu. 2024.
Alternative zkLLM design using folding techniques and Incrementally Verifiable Computation (IVC). Compares IVC constructions based on SNARKs vs. folding schemes.
[ePrint 2024/480](https://eprint.iacr.org/2024/480)

**ZKML: An Optimizing System for ML Inference in Zero-Knowledge Proofs**
Kang et al. EuroSys 2024.
Framework for producing ZK-SNARKs for realistic ML models including distilled GPT-2 and Twitter's recommendation model.
[ACM DL](https://dl.acm.org/doi/10.1145/3627703.3650088)

**zkCNN: Zero Knowledge Proofs for Convolutional Neural Network Predictions and Accuracy**
Liu, Xie, Zhang. ACM CCS 2021.
Proves CNN inference without leaking model information. Introduces a new sumcheck protocol for FFTs and convolutions with linear prover time.
[ePrint 2021/673](https://eprint.iacr.org/2021/673) | [GitHub](https://github.com/TAMUCrypto/zkCNN)

**ZEN: An Optimizing Compiler for Verifiable, Zero-Knowledge Neural Networks**
Feng, Qin et al. 2021.
First optimizing compiler for verifiable neural networks with zero-knowledge. Addresses floating-point to finite-field conversion.
[ePrint 2021/087](https://eprint.iacr.org/2021/087.pdf)

**Scaling up Trustless DNN Inference with Zero-Knowledge Proofs**
Kang, Hashimoto, Stoica, Sun. 2022.
First practical ImageNet-scale method to verify ML model inference non-interactively using ZK-SNARKs.
[arXiv](https://arxiv.org/abs/2210.08674)

**vCNN: Verifiable Convolutional Neural Network based on zk-SNARKs**
Lee et al. 2020.
One of the earliest works applying zk-SNARKs specifically to CNN inference verification.
[ePrint 2020/584](https://eprint.iacr.org/2020/584.pdf)

**Zero-Knowledge Proof Based Verifiable Inference of Models**
Wang et al. 2025.
ZK framework for verifying deep learning inference. Claims scalability to DeepSeek-V3 (671B parameters) with constant proof size.
[arXiv](https://arxiv.org/abs/2511.19902)

**zkFinGPT: Zero-Knowledge Proofs for Financial Generative Pre-trained Transformers**
Liu et al. 2026.
Applies ZKPs to financial LLM use cases including copyright disputes and trading strategy protection.
[arXiv](https://arxiv.org/abs/2601.15716)

**Verifiable Evaluations of Machine Learning Models using zkSNARKs**
South et al. 2024.
Verifiable model evaluation producing zero-knowledge computational proofs packaged into verifiable evaluation attestations.
[arXiv](https://arxiv.org/abs/2402.02675)

**TeleSparse: Practical Privacy-Preserving Verification of Deep Neural Networks**
2025. PETS 2025.
ZK-friendly post-processing using sparsification and neural teleportation to reduce circuit constraints.
[arXiv](https://arxiv.org/abs/2504.19274) | [GitHub](https://github.com/mammadmaheri7/TeleSparseRepo)

**An Efficient and Extensible Zero-knowledge Proof Framework for Neural Networks**
2024.
Uses Vector Oblivious Linear Evaluation (VOLE) as proving backend for large-scale neural network proof generation.
[ePrint 2024/703](https://eprint.iacr.org/2024/703.pdf)

**Kaizen: Zero-Knowledge Proofs of Training for Deep Neural Networks**
Abbaszadeh, Pappas et al. ACM CCS 2024.
Zero-knowledge proof of training for DNNs with provable security/privacy and succinct proof size.
[ePrint 2024/162](https://eprint.iacr.org/2024/162)

**zkDL: Efficient Zero-Knowledge Proofs of Deep Learning Training**
Sun et al. 2023.
Introduces zkReLU for ReLU activation proofs and FAC4DNN for converting DL to arithmetic circuits. First ZKML platform on GPUs.
[arXiv](https://arxiv.org/abs/2307.16273) | [GitHub](https://github.com/SafeAILab/zkDL)

---

## 2. zkML Frameworks and Code Repositories

**EZKL (Zkonduit)**
Engine for doing inference for deep learning models in zk-SNARKs. Converts ONNX models to Halo2 circuits automatically.
[GitHub](https://github.com/zkonduit/ezkl) | [Docs](https://docs.ezkl.xyz/) | [Benchmarks](https://blog.ezkl.xyz/post/benchmarks/)

**RISC Zero zkVM**
General-purpose zkVM based on zk-STARKs and RISC-V. Supports ML inference via SmartCore ML framework. No quantization needed.
[GitHub](https://github.com/risc0/risc0) | [Website](https://risczero.com/)

**Giza / Orion**
ZKML framework using Cairo language for Starknet blockchain. Transpiles ONNX models.
[GitHub](https://github.com/gizatechxyz) | [Website](https://www.gizatech.xyz/)

**Zator / Zator2**
Verified inference of a 512-layer CNN using Nova recursive SNARKs.
[GitHub](https://github.com/lyronctk/zator) | [GitHub (v2)](https://github.com/zero-savvy/zator2)

**awesome-zkml (Worldcoin/World)**
Curated list of ZKML projects, papers, tools, and resources.
[GitHub](https://github.com/worldcoin/awesome-zkml)

**zkml-blueprints (Inference Labs)**
Mathematical formulations and circuit designs for zero-knowledge proofs in ML applications.
[GitHub](https://github.com/inference-labs-inc/zkml-blueprints)

**Modulus Labs (now part of Tools for Humanity / World)**
Published "The Cost of Intelligence" benchmarking ZK proof systems against ML models. First to prove a multi-billion parameter LLM in ZK.
[World blog](https://world.org/blog/announcements/modulus-labs-joins-tfh-support-applied-research-world) | [Variant investment](https://variant.fund/articles/modulus-zero-knowledge-machine-learning-seed-round/)

---

## 3. Steganography, Covert Channels, and Information Hiding in LLMs

**An Information-Theoretic Model for Steganography**
Cachin. 1998/2004.
The foundational paper defining information-theoretic security for steganography: security is quantified by the KL divergence between cover and stego distributions.
[ePrint](https://eprint.iacr.org/2000/028)

**Perfectly Secure Steganography Using Minimum Entropy Coupling**
Schroeder de Witt, Sokota, Kolter, Foerster, Strohmeier. ICLR 2023.
Proves a steganographic procedure is perfectly secure iff induced by a coupling, and maximizes throughput iff induced by a minimum entropy coupling.
[arXiv](https://arxiv.org/abs/2210.14889)

**Meteor: Cryptographically Secure Steganography for Realistic Distributions**
Kaptchuk, Jois, Green, Rubin. ACM CCS 2021.
Symmetric-key steganography protocol that adapts encoding rate to local entropy of the text generation model.
[ACM DL](https://dl.acm.org/doi/10.1145/3460120.3484550)

**Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs**
Mathew, Matthews, McCarthy, Velja, Schroeder de Witt, Cope, Schoots. NeurIPS 2024.
Demonstrates unintended steganographic collusion can arise from misspecified reward incentives during training.
[arXiv](https://arxiv.org/abs/2410.03768)

**The Steganographic Potentials of Language Models**
Karpov, Adeleke, Cho, Perez-Campanero. ICLR 2025 Workshop.
Explores steganographic capabilities of LLMs fine-tuned via RL.
[arXiv](https://arxiv.org/abs/2505.03439)

**Robust Steganography from Large Language Models**
Perry, Gupte, Pitta, Rotem. 2025.
Introduces formal definitions of weak and strong robust LLM-based steganography.
[arXiv](https://arxiv.org/abs/2504.08977)

**Generative Text Steganography with Large Language Model (LLM-Stega)**
ACM MM 2024.
Black-box generative text steganographic method using LLM user interfaces.
[arXiv](https://arxiv.org/abs/2404.10229)

**L^2 * M = C^2: Large Language Models Are Covert Channels**
Gaure, Koffas, Picek, Ronjom. IEEE, 2024.
Empirically measures security vs. capacity of open-source LLMs as covert channels, finding ~1 bit per word average bitrate.
[arXiv](https://arxiv.org/abs/2405.15652)

**Look Who's Talking Now: Covert Channels From Biased LLMs**
Silva, Sala, Gabrys. EMNLP 2024 Findings.
Adapts LLM watermarking strategies to encode large amounts of information, studies fundamental limits of steganographic insertion.
[ACL Anthology](https://aclanthology.org/2024.findings-emnlp.971/)

**Encryption Based Covert Channel for Large Language Models**
2024.
Encryption-based covert channel construction using LLMs with cryptographic security guarantees.
[ePrint 2024/586](https://eprint.iacr.org/2024/586)

**Discop: Provably Secure Steganography in Practice**
Ding et al. IEEE S&P 2023.
Practical provably secure steganography system building on autoregressive language models.
[PDF](https://dingjinyang.github.io/uploads/Discop_sp23_paper.pdf)

**Early Signs of Steganographic Capabilities in Frontier LLMs**
Zolkowski et al. 2025.
Investigates whether frontier LLMs already show signs of steganographic capabilities without being explicitly trained for it.
[arXiv](https://arxiv.org/abs/2507.02737)

---

## 4. Model Weight Exfiltration and Inference Verification

**Verifying LLM Inference to Detect Model Weight Exfiltration**
Rinberg, Karvonen, Hoover, Reuter, Warr. 2025.
Formalizes model weight exfiltration as a security game. Limits steganographic exfiltration to <0.5% of total information, extending exfiltration timelines by over 200x.
[arXiv](https://arxiv.org/abs/2511.02620) | [GitHub](https://github.com/RoyRin/inference_verification_for_model_weight_exfiltration)

**DiFR: Inference Verification Despite Nondeterminism**
Karvonen et al. 2025.
Token-DiFR verifies inference outputs by comparing generated tokens against a trusted reference. Activation-DiFR uses random orthogonal projections to compress activations into compact fingerprints.
[arXiv](https://arxiv.org/abs/2511.20621) | [Blog](https://adamkarvonen.github.io/machine_learning/2025/11/28/difr.html)

**Using an LLM Perplexity Filter to Detect Weight Exfiltration**
Karvonen. 2024.
Proposes perplexity-based filtering to distinguish normal model outputs from steganographic weight exfiltration attempts.
[Blog](https://adamkarvonen.github.io/machine_learning/2024/07/21/weight-exfiltration.html)

**How Does LLM Compression Affect Weight Exfiltration Attacks?**
Brown. 2025.
Studies the interaction between model compression techniques and the feasibility of weight exfiltration attacks.
[PDF](https://openreview.net/pdf/6224e8f8a640d660f25a1d0d7ca20f6179816879.pdf)

**Defending Against Model Weight Exfiltration Through Inference Verification**
2025.
Discussion of defenses against steganographic exfiltration, noting LLM inference is nearly deterministic (~98% of tokens match on re-generation).
[LessWrong](https://www.lesswrong.com/posts/7i33FDCfcRLJbPs6u/defending-against-model-weight-exfiltration-through-1)

---

## 5. GPU Floating-Point Non-Determinism

**On the Structure of Floating-Point Noise in Batch-Invariant GPU Matrix Multiplication**
Yashwanth et al. 2025.
Demonstrates that floating-point errors in GPU matmul are not i.i.d. Gaussian but structured and highly correlated.
[arXiv](https://arxiv.org/abs/2511.00025)

**Impacts of Floating-Point Non-Associativity on Reproducibility for HPC and Deep Learning Applications**
Shanmugavelu et al. 2024.
Examines how floating-point non-associativity leads to large variations in deep learning inference pipelines.
[arXiv](https://arxiv.org/abs/2408.05148)

**MMA-Sim: Bit-Accurate Reference Model of Tensor Cores and Matrix Cores**
2025.
Bit-accurate reference model for NVIDIA Tensor Cores and AMD Matrix Cores.
[arXiv](https://arxiv.org/abs/2511.10909)

**Solving Reproducibility Challenges in Deep Learning and LLMs: Our Journey (Ingonyama)**
2025.
Rewrote GEMM CUDA kernels for llama.cpp to be deterministic by avoiding Tensor Cores. Performance impact primarily in prompt processing.
[Blog](https://www.ingonyama.com/post/solving-reproducibility-challenges-in-deep-learning-and-llms-our-journey)

**Defeating Nondeterminism in LLM Inference**
2025.
Practical approaches to achieving deterministic LLM inference.
[Blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

**Deterministic Inference Mode for llama.cpp**
PR #16016. 2025.
Adds opt-in deterministic mode making CUDA inference bit-identical for identical inputs.
[GitHub PR](https://github.com/ggml-org/llama.cpp/pull/16016)

---

## 6. AI Governance, Compute Verification, and International Agreements

**Computing Power and the Governance of Artificial Intelligence**
Sastry, Heim, Belfield, Anderljung, Brundage et al. 2024.
Argues compute is a particularly effective governance intervention point because it is detectable, excludable, and quantifiable.
[arXiv](https://arxiv.org/abs/2402.08797)

**Verification Methods for International AI Agreements**
Wasil, Reed, Miller, Barnett. 2024.
Examines 10 verification methods for detecting unauthorized AI training and data centers.
[arXiv](https://arxiv.org/abs/2408.16074)

**Verifying International Agreements on AI: Six Layers of Verification**
Baker, Kulp, Marks, Brundage, Heim. RAND, 2025.
Proposes six largely independent verification layers including chip security features, network taps, and analog sensors.
[RAND](https://www.rand.org/pubs/working_papers/WRA4077-1.html) | [arXiv](https://arxiv.org/abs/2507.15916)

**UN Scientific Advisory Board Brief: Verification of Frontier AI**
2025.
UN brief on verifying frontier AI systems.
[PDF](https://www.un.org/scientific-advisory-board/sites/default/files/2025-06/verification_of_frontier_ai.pdf)

**Open Problems in Technical AI Governance**
Reuel et al. Stanford / GovAI.
Survey of open technical problems in AI governance, including compute monitoring and verification.
[PDF](https://cdn.governance.ai/Open_Problems_in_Technical_AI_Governance.pdf)

**Governing Through the Cloud: The Intermediary Role of Compute Providers in AI Regulation**
GovAI.
Analyzes how cloud service providers can serve as intermediaries for AI governance.
[PDF](https://cdn.governance.ai/Governing-Through-the-Cloud_The-Intermediary-Role-of-Compute-Providers-in-AI-Regulation.pdf)

**Strategies and Detection Gaps in a Game-Theoretic Model of Compute Governance**
Moon, Vedula, Geneson, Bar-on. RAND, 2025.
Finds that FLOP-threshold-based monitoring has significant detection gaps.
[RAND](https://www.rand.org/pubs/research_reports/RRA3686-1.html)

**Faster AI Diffusion Through Hardware-Based Verification**
Institute for Progress.
Proposes hardware design enabling cryptographic proof of claims about AI development and usage.
[IFP](https://ifp.org/faster-ai-diffusion-through-hardware-based-verification/)

**Framework for End-to-End Verifiable AI Pipelines**
2025.
Proposes components required for verifiability across the entire AI pipeline.
[arXiv](https://arxiv.org/html/2503.22573v1)

---

## 7. Hardware Attestation and Trusted Execution Environments

**SAGE: Software-based Attestation for GPU Execution**
Ivanov, Rothenberger, Dethise, Canini, Hoefler, Perrig. USENIX ATC 2023.
Software-based attestation for NVIDIA A100 GPUs providing code integrity, secrecy, and computation integrity.
[USENIX](https://www.usenix.org/conference/atc23/presentation/ivanov)

**DeepAttest: An End-to-End Attestation Framework for Deep Neural Networks**
Chen et al. ISCA 2019.
First on-device DNN attestation method embedding a device-specific fingerprint in model weights.
[ACM DL](https://dl.acm.org/doi/10.1145/3307650.3322251)

**Laminator: Verifiable ML Property Cards using Hardware-assisted Attestations**
Duddu, Gunn et al. ACM CODASPY 2025.
Uses TEEs to create verifiable ML property cards with bindings between training dataset properties and models.
[arXiv](https://arxiv.org/abs/2406.17548)

**Attestable Audits: Verifiable AI Safety Benchmarks Using Trusted Execution Environments**
Schnabl et al. ICML 2025.
Protocol where auditors and model providers load models/benchmarks into TEEs, run benchmarks, and cryptographically attest results.
[arXiv](https://arxiv.org/abs/2506.23706)

**Secured and Privacy-Preserving GPU-Based Machine Learning Inference in Trusted Execution Environment: A Comprehensive Survey**
2025.
Survey covering GPU-based ML inference in TEEs, including NVIDIA H100 confidential computing.
[IEEE](https://ieeexplore.ieee.org/document/10885734/)

---

## 8. LLM Watermarking and Entropy

**A Watermark for Large Language Models**
Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein. ICML 2023.
Foundational LLM watermarking paper using "green list" token promotion. Watermark effectiveness depends directly on output entropy.
[arXiv](https://arxiv.org/abs/2301.10226)

**Scalable Watermarking for Identifying Large Language Model Outputs**
Nature, 2024.
Scalable watermarking technique with formal analysis of the entropy-watermark relationship.
[Nature](https://www.nature.com/articles/s41586-024-08025-4)

**Undetectable Watermarks for Language Models**
Christ, Gunn, Zamir. COLT 2024.
Constructs cryptographically undetectable watermarks based on one-way functions.
[arXiv](https://arxiv.org/abs/2306.09194)

**Invisible Entropy: Towards Safe and Efficient Low-Entropy LLM Watermarking**
2025.
Addresses watermarking low-entropy outputs by removing reliance on the original LLM for entropy calculations.
[arXiv](https://arxiv.org/abs/2505.14112)

**A Survey of Text Watermarking in the Era of Large Language Models**
ACM Computing Surveys, 2024.
Comprehensive survey covering the full landscape of text watermarking methods.
[ACM DL](https://dl.acm.org/doi/10.1145/3691626)

---

## 9. Zero-Knowledge Proof Primitives and Cryptography

### Sum-Check Protocol

**Algebraic Methods for Interactive Proof Systems**
Lund, Fortnow, Karloff, Nisan. STOC 1990 / JACM 1992.
The original paper introducing the sum-check protocol.
[Wikipedia](https://en.wikipedia.org/wiki/Sum-check_protocol)

**The Unreasonable Power of the Sum-Check Protocol**
Thaler. 2020.
How the sum-check protocol underpins modern zk-SNARKs.
[ZKProof](https://zkproof.org/2020/03/16/sum-checkprotocol/) | [PDF](https://people.cs.georgetown.edu/jthaler/blogpost.pdf)

**Sum-check Is All You Need: An Opinionated Survey on Fast Provers in SNARK Design**
2025.
Argues that fast SNARKs converge on the sum-check protocol as the central component.
[ePrint 2025/2041](https://eprint.iacr.org/2025/2041)

### GKR Protocol

**Delegating Computation: Interactive Proofs for Muggles**
Goldwasser, Kalai, Rothblum. STOC 2008 / JACM 2015.
The original GKR protocol paper for interactive proofs on layered arithmetic circuits.
[ACM DL](https://dl.acm.org/doi/10.1145/3476446.3536183)

**Time-Optimal Interactive Proofs for Circuit Evaluation**
Thaler. 2013.
Achieves optimal O(S log S) prover time for the GKR protocol.
[ePrint 2013/351](https://eprint.iacr.org/2013/351.pdf)

### Lookup Arguments

**plookup: A Simplified Polynomial Protocol for Lookup Tables**
Gabizon, Williamson. 2020.
Foundational polynomial-based method for proving values belong to a pre-defined table.
[ePrint 2020/315](https://eprint.iacr.org/2020/315.pdf)

**Caulk: Lookup Arguments in Sublinear Time**
Zapico, Buterin, Khovratovich, Maller, Nitulescu, Simkin. ACM CCS 2022.
First lookup argument with prover time sublinear in the table size.
[ePrint 2022/621](https://eprint.iacr.org/2022/621)

**CQ: Cached Quotients for Fast Lookups**
Eagen, Fiore, Gabizon. 2023.
Reduces prover and preprocessing runtimes using log-derivative methods and precomputed cached quotients.
[Springer](https://link.springer.com/article/10.1007/s00145-024-09535-0)

### Pedersen Commitments

**Non-Interactive and Information-Theoretic Secure Verifiable Secret Sharing**
Pedersen. 1991.
The original Pedersen commitment scheme — information-theoretically hiding and computationally binding.

**Polynomial Commitments (KZG)**
Kate, Zaverucha, Goldberg. 2010.
KZG polynomial commitment scheme widely used in modern SNARKs.
[PDF](https://cacr.uwaterloo.ca/techreports/2010/cacr2010-10.pdf)

### Multilinear Extensions

**Zeromorph: Zero-Knowledge Multilinear-Evaluation Proofs from Homomorphic Univariate Commitments**
Kohrita, Tanner. 2023.
Efficient scheme for committing to multilinear polynomials and proving evaluations with zero-knowledge.
[ePrint 2023/917](https://eprint.iacr.org/2023/917)

**DeepFold: Efficient Multilinear Polynomial Commitment from Reed-Solomon Code**
2024.
Efficient multilinear polynomial commitments without pairings.
[ePrint 2024/1595](https://eprint.iacr.org/2024/1595.pdf)

### Range Proofs

**SoK: Zero-Knowledge Range Proofs**
Christ, Baldimtsi, Chalkias, Maram, Roy, Wang. AFT 2024.
Systematization of knowledge covering all major ZKRP constructions.
[ePrint 2024/430](https://eprint.iacr.org/2024/430.pdf)

**Bulletproofs: Short Proofs for Confidential Transactions and More**
Bunz, Bootle, Boneh, Poelstra, Wuille, Maxwell. IEEE S&P 2018.
Logarithmic-size range proofs using inner-product arguments with Pedersen vector commitments.

### Proof Systems

**Libra: Succinct Zero-Knowledge Proofs with Optimal Prover Computation**
Xie, Zhang, Zhang, Papamanthou, Song. CRYPTO 2019.
First ZK proof system achieving both optimal O(C) prover time and succinct proof size.
[PDF](https://www.cs.yale.edu/homes/cpap/published/libra-crypto19.pdf) | [GitHub](https://github.com/sunblaze-ucb/Libra)

**Linear-Time Zero-Knowledge Proofs for Arithmetic Circuit Satisfiability**
Bootle, Cerulli, Ghadafi, Groth, Hajiabadi, Jakobsen. 2017.
Achieves ZK proofs with prover running in linear time in circuit size.
[ePrint 2017/872](https://eprint.iacr.org/2017/872.pdf)

### BLS12-381 Curve

**BLS12-381 For The Rest Of Us**
Edgington.
Comprehensive technical explainer of the BLS12-381 pairing-friendly curve.
[HackMD](https://hackmd.io/@benjaminion/bls12-381)

**zkcrypto/bls12_381**
Rust implementation widely used in ZK proof implementations.
[GitHub](https://github.com/zkcrypto/bls12_381)

---

## 10. Textbooks and Surveys

**Proofs, Arguments, and Zero-Knowledge**
Thaler. 2023. Foundations and Trends in Privacy and Security.
The definitive textbook covering interactive proofs, the sumcheck protocol, GKR, polynomial commitments, and ZK argument design. Freely available.
[PDF](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)

**A Survey of Zero-Knowledge Proof Based Verifiable Machine Learning**
2025.
Comprehensive survey of all ZKML research from June 2017 to December 2024.
[arXiv](https://arxiv.org/abs/2502.18535)

**On-chain Zero-Knowledge Machine Learning: An Overview and Comparison**
2024.
Overview comparing EZKL, RISC Zero, and Giza Orion for blockchain-based verifiable ML inference.
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157824002969)

**A Survey on Zero Knowledge Range Proofs and Applications**
Morais, Koens, van Wijk, Koren. 2019.
Survey covering range proof techniques and their applications in privacy-preserving systems.
[arXiv](https://arxiv.org/pdf/1907.06381)

---

## 11. Blog Posts and Educational Resources

**The Definitive Guide to ZKML (2025)** — ICME
[Blog](https://blog.icme.io/the-definitive-guide-to-zkml-2025/)

**An Introduction to Zero-Knowledge Machine Learning (ZKML)** — World (Worldcoin)
[Blog](https://world.org/blog/engineering/intro-to-zkml)

**The State of Zero-Knowledge Machine Learning (zkML)** — Spectral Finance
[Blog](https://blog.spectral.finance/the-state-of-zero-knowledge-machine-learning-zkml/)

**ZKML: Verifiable Machine Learning using Zero-Knowledge Proof** — Kudelski Security
[Blog](https://kudelskisecurity.com/modern-ciso-blog/zkml-verifiable-machine-learning-using-zero-knowledge-proof)

**If you don't know, look it up** — LambdaClass (lookup arguments tutorial)
[Blog](https://blog.lambdaclass.com/lookups/)

**GKR Protocol: A Step-by-Step Example** — LambdaClass
[Blog](https://blog.lambdaclass.com/gkr-protocol-a-step-by-step-example/)

**The Lookup Singularity** — ICME
[Blog](https://blog.icme.io/the-lookup-singularity/)

**What are Pedersen Commitments and How They Work** — RareSkills
[Blog](https://rareskills.io/post/pedersen-commitment)

**Arithmetic Circuits for ZK** — RareSkills
[Blog](https://rareskills.io/post/arithmetic-circuit)

**Sum-Check Protocol and Multilinear Extensions** — RisenCrypto
[Blog](https://risencrypto.github.io/Sumcheck/)

**Georgetown University Course Notes on the Sum-Check Protocol** — Thaler
[PDF](https://people.cs.georgetown.edu/jthaler/sumcheck.pdf)

**Modern Zero Knowledge Cryptography** — MIT IAP Course, January 2023
[PDF](https://assets.super.so/9c1ce0ba-bad4-4680-8c65-3a46532bf44a/files/61fb28e6-f2dc-420f-89e1-cc8000233a4f.pdf)

**Berkeley RDI Zero-Knowledge Proof Research**
[Website](https://rdi.berkeley.edu/zkp/)

**Crucial Considerations for Compute Governance** — Heim
[Blog](https://blog.heim.xyz/crucial-considerations-for-compute-governance/)
