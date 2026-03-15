# Zero Knowledge Conditional Entropy Bounds for LLM Inference Verification
## With Probabilistic Tolerance for Hardware Non-Determinism

This repository forks [zkLLM](https://arxiv.org/abs/2404.16109) and adds a zero knowledge proof of conditional entropy for LLM inference outputs.

## What is inference verification?

Verifying that a token stream is consistent with inference by a claimed model on a claimed input — like model fingerprinting, but with a slightly different goal: we are not trying to rule out other ways of generating the text.

## Why verify inference?

- Find bugs in an inference stack
- Verify that the model purchased is the same one that is served
- Verify that regulator evals run on the correct model
- Verify that model weights aren't being secretly exfiltrated in inference results
- Verify that compute is being used as claimed (potentially for international agreements about AI)

## Why do this in zero knowledge?

- Enables verification of closed-weight models where the verifier may not have access to model weights
- Protects against model weight theft even if the verifier has model access
- Side benefit: formal security guarantees that don't rely on a complex software and hardware stack

## What was missing before this project

Existing work (e.g. zkLLM, zkML, [this paper](https://arxiv.org/pdf/2511.02620), and [this paper](https://arxiv.org/abs/2511.20621)) either verifies exact inference on finite fields (not using floating point tensor cores), or classifies whether a token stream is consistent with a claimed model without a zero knowledge guarantee. There was no zero knowledge verification of efficient floating point inference — to use prior ZK approaches you would have had to fully rewrite the inference stack to use finite fields instead of floating point.

## Contributions

- Reframed the problem as bounding conditional entropy in the output (a quantity, not a classification)
- Showed that arbitrary approximations of token probability provide a strict upper bound on conditional entropy (Gibbs' inequality)
- Developed a simple formula for token probabilities that accounts for hardware noise: normal distribution with difference in logits between the token and the max token
- Prototyped these modifications by forking zkLLM and adding a zero knowledge proof of conditional entropy

## Conclusion

- Prototyped a zero knowledge proof of conditional entropy
- Demonstrated feasibility of zero knowledge inference verification without reducing floating point inference efficiency
- (~1000x slower to generate the proof than to do inference, but this can be done on a small random sample to amortize the cost)
