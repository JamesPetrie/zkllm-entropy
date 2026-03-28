

What is inference verification?
- verifying that a token stream is consistent with inference by a claimed model on a claimed input
- (like model fingerprinting, but a slightly different goal because we’re not trying to rule out other ways of generating the text)


Why verify inference?
- find bugs in inference stack
- verify that model purchased is the same one that is served 
- verify that regulator evals run on the correct model
- verify that model weights aren’t being secretly exfiltrated in inference results
- verify that compute is being used as claimed (potentially for international agreements about AI)


Why do this in zero knowledge?
- verification of closed weight models
    - The verifier might not have access to model weights
    - protect against model weight theft (even if the verifier has model access)
- side benefit: formal guarantees of security that don’t rely on complex software and hardware stack


Existing work:
- zkllm, zkml and other zero knowledge projects verifying exact inference evaluated on finite field (not using floating point tensor cores)
- https://arxiv.org/pdf/2511.02620 and https://arxiv.org/abs/2511.20621 : papers that present algorithms to classify whether a token stream is consistent with the claimed model



What was missing before this project:
- zero knowledge verification of efficient floating point inference
- (would have had to fully rewrite inference stack to use finite field instead of floating point)


Contributions: 
- reframed problem as bounding conditional entropy in the output (quantity, not classification)
- Showed that arbitrary approximations of token probability provide a strict upper bound on conditional entropy (Gibbs’ inequality)
- Developed a simple formula for token probabilities that accounts for hardware noise: normal distribution with difference in logits between the token and the max token
- Prototyped these modifications by forking ZKLLM and adding a zero knowledge proof of conditional entropy: https://github.com/JamesPetrie/zkllm-entropy 



Conclusion:
- prototyped zero knowledge proof of conditional entropy: https://github.com/JamesPetrie/zkllm-entropy 
- demonstrated feasibility of zero knowledge inference verification without reducing floating point inference efficiency 
- (~1000x slower to generate the proof than do inference, but this can be done on a small random sample to amortize the cost)

