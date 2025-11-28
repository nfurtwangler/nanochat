# Partial Collapse for Autoregressive Language Models
### A Proposed Method for Propagating Uncertainty Across Time Steps

**Author:** (brandf, nfurtwangler)  
**Summary prepared by:** ChatGPT  

---

## 1. Motivation

Standard autoregressive language models commit to a *single token* at each generation step.  
This “hard collapse” converts the probability distribution over vocab tokens into a one‑hot vector, which:

- **destroys uncertainty information**
- **prevents the model from knowing how close other candidate tokens were**
- **contributes to exposure bias**, because once the model samples an uncertain token, all future inputs are conditioned on that discrete decision
- **makes the entire past artificially “certain”**, even when the model's logits were highly ambiguous

However, carrying forward the *entire* probability distribution (e.g., in embedding space) is computationally expensive and may destabilize models trained on one‑hot token inputs.

**Partial collapse** is a compromise that:

1. Retains the benefits of discrete sampling  
2. Preserves useful uncertainty information  
3. Avoids the cost of full distribution embedding  
4. Can be trained directly using teacher forcing  
5. Uses a top‑k approximation to keep computation small

---

## 2. Core Idea: Partial Collapse Embedding

At decoding time:

1. The model outputs a probability distribution  
   \[
   p_t = \text{softmax}(z_t)
   \]

2. A token \(k_t\) is sampled from this distribution.

3. Instead of feeding the next layer the hard embedding \(E_{k_t}\), we construct a **partial collapse embedding**:

### **Definition**
For collapse factor \(\alpha \in [0,1]\):

\[
x_{t+1}
= \alpha E_{k_t}
+ (1-\alpha) \sum_{i \in \text{top-k}(p_t)} \hat{p}_{t,i} E_i
\]

Where:

- \(E_i\) = embedding of token \(i\)  
- \(\text{top-k}(p_t)\) = k highest-probability tokens  
- \(\hat{p}_{t,i}\) = probabilities renormalized over this top‑k slice

### **Interpretation**
- \(\alpha E_{k_t}\): keeps the generation path grounded in actual sampled tokens  
- \((1-\alpha)\) mixture: injects uncertainty information into the next step’s input  
- Using only top‑k tokens avoids expensive full \(V \times d\) matmuls

Typical choices:

- \(\alpha \in [0.8, 0.95]\)  
- \(k \in [10, 64]\)

---

## 3. Efficient Partial Collapse (Top‑k)

### Why top‑k?
The expected embedding:

\[
\sum_i p_i E_i
\]

is expensive — O(V·d).  
But most of the mass usually lies in the top‑k tokens.

Thus we approximate:

\[
\sum_i p_i E_i \approx \sum_{i \in S} \hat{p}_{i} E_i
\]

Where:

- \(S\) = top-k slice  
- \(\hat{p}\) = renormalized distribution over S  

This reduces computation to **O(k·d)**.

---

## 4. Dual-Pass Training to Avoid Distribution Shift

During training, the model only ever sees *hard* embeddings.  
But at inference, we're feeding *continuous mixtures*.  
Without alignment, this distribution shift can produce degradation.

### Solution: A Two-Pass Training Step

For each minibatch:

---

### **Pass 1 — Standard Teacher Forcing**
Inputs:  
\[
E_{y_0}, E_{y_1}, \dots, E_{y_{T-1}}
\]

Outputs:  
\[
p_t = \text{softmax}(z_t)
\]

Loss:  
\[
L_{\text{hard}} = \sum_t \text{CE}(p_t, y_t)
\]

Detach probabilities so the second pass does not backprop through the first:  
\[
p_t^\* = \text{stop\_grad}(p_t)
\]

---

### **Construct Partial-Collapse Inputs**
For each position \(t > 0\):

1. Extract top‑k tokens of \(p_{t-1}^\*\)  
2. Renormalize their probabilities  
3. Construct:

\[
x_t^{(\text{pc})}
= \alpha E_{y_{t-1}}
+ (1-\alpha)\sum_{i \in \text{top-k}} \hat{p}_{t-1,i} E_i
\]

Use BOS for \(t = 0\).

---

### **Pass 2 — Teacher Forcing with Partial-Collapse Inputs**
Inputs:  
\[
x_0, x_1^{(\text{pc})}, \dots, x_T^{(\text{pc})}
\]

Outputs:  
\[
\tilde{p}_t = \text{softmax}(\tilde{z}_t)
\]

Loss:  
\[
L_{\text{pc}} = \sum_t \text{CE}(\tilde{p}_t, y_t)
\]

---

### **Final Training Loss**
\[
L = L_{\text{hard}} + \lambda L_{\text{pc}}
\]

Where \(\lambda \in [0.1, 0.5]\).

This teaches the model:

- how to interpret uncertainty-bearing mixed embeddings  
- how to behave stably under partial-collapse inputs  
- how to keep predictions invariant across hard vs soft inputs

---

## 5. Inference Procedure (Runtime Decoding)

Given model output distribution \(p_t\):

1. Sample token \(k_t\)
2. Get top‑k slice of \(p_t\)
3. Compute the partial-collapse embedding
4. Feed this mixed latent into the model as the next input
5. Repeat

The algorithm is identical to the training partial‑collapse embedding mechanism, except that ground‑truth tokens are replaced by sampled tokens.

---

## 6. Hyperparameters

Recommended defaults:

| Parameter | Typical Values | Notes |
|----------|----------------|-------|
| \(\alpha\) | 0.85–0.95 | Higher = more discrete, stable |
| \(k\) | 20–50 | Larger = more uncertainty encoded |
| \(\lambda\) | 0.2 | Weight for partial-collapse loss |
| Schedule for \(\alpha\) | Optional | Slight annealing downward during training may help |
| Stop-grad on first pass | **Required** | Stabilizes dual-pass behavior |

---

## 7. Variants and Extensions

### 7.1 Single-pass mixed teacher forcing
Randomly replace some positions with partial-collapse embeddings rather than running two full passes.

### 7.2 KL-consistency regularization
Instead of a 2nd CE loss, enforce:

\[
\mathrm{KL}(p_t^{\text{hard}} \parallel p_t^{\text{pc}})
\]

### 7.3 Scheduled Sampling Analogue
Increase the model's dependence on its own soft predictions over training time by decaying \(\alpha\).

---

## 8. Expected Benefits

- **Propagates uncertainty** rather than destroying it each step  
- **Reduces exposure bias**  
- **Produces more stable long-range generation**  
- **Enables smoother control over diversity**  
- **Adds minimal overhead** with top‑k approximation  
- **Compatible with existing transformer architectures**  
- **Can be trained from scratch or fine‑tuned**  

---

## 9. Implementation Outline (Pseudo-Code)

```python
for batch in dataset:

    # --- Pass 1: Hard teacher forcing ---
    logits = model(hard_inputs)                # embeddings of ground-truth tokens
    p = softmax(logits)
    L_hard = cross_entropy(p, targets)

    p_detached = stop_gradient(p)

    # --- Build partial-collapse embeddings ---
    x_pc = []
    for t in range(seq_len):
        if t == 0:
            x_pc.append(bos_embedding)
        else:
            topk_vals, topk_idx = topk(p_detached[t-1], k)
            probs = normalize(topk_vals)
            soft_part = sum(probs[i] * embedding(topk_idx[i]) for i in range(k))
            mixed = alpha * embedding(targets[t-1]) + (1-alpha) * soft_part
            x_pc.append(mixed)

    # --- Pass 2: Teacher forcing with mixed latents ---
    logits_pc = model(x_pc)
    p_pc = softmax(logits_pc)
    L_pc = cross_entropy(p_pc, targets)

    # --- Combine losses ---
    L = L_hard + lambda_ * L_pc
    L.backward()
    optimizer.step()
