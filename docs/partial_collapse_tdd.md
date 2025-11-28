# Technical Design Document (TDD)
## Feature: Partial Collapse Decoding + Dual-Pass Training
### Version: 1.0
### Status: Draft
### Author: brandf, nfurtwangler
### Prepared by: ChatGPT

---

# 1. Overview

This TDD defines the engineering requirements, algorithms, data flow, and interfaces needed to implement **Partial Collapse** in an autoregressive transformer language model.

Partial Collapse is a decoding and training mechanism that:

- Uses a convex mixture of the **sampled token embedding** and a **top‑k expected embedding**  
- Propagates uncertainty across time steps  
- Reduces exposure bias  
- Minimizes extra compute by avoiding full vocabulary embedding operations  
- Uses a **dual-pass training procedure** to eliminate training/inference distribution shift

This document specifies exactly what code needs to be written, how modules interact, and what behaviors must be testable.

---

# 2. Requirements

## 2.1 Functional Requirements

### F1 — Partial Collapse Decoding
The model shall:
1. Produce token distributions \(p_t\) at each step.
2. Sample token \(k_t\) from \(p_t\).
3. Compute a mixed embedding:
   \[
   x_{t+1} = \alpha E_{k_t} + (1-\alpha)\sum_{i \in \text{top-k}} \hat{p}_{t,i} E_i
   \]
4. Feed this embedding into the next transformer block.

### F2 — Top‑k Approximation
- Only the top‑k tokens in \(p_t\) are used in the mixture.
- Their probabilities are renormalized.

### F3 — Dual-Pass Training
Training shall involve:
- **Pass 1:** Standard teacher forcing  
- **Pass 2:** Teacher forcing using partial-collapse embeddings constructed from Pass 1 predictions  

### F4 — Loss Combination
\[
L = L_{\text{hard}} + \lambda L_{\text{pc}}
\]

### F5 — Stop‑Gradient
- Probabilities used to construct partial-collapse embeddings must be gradient‑detached.

---

## 2.2 Nonfunctional Requirements

### N1 — Efficiency
- Additional compute per step must be **O(k · d)**, not **O(V · d)**.
- Avoid duplicating unnecessary tensors between passes.

### N2 — Deterministic Behavior
- Given fixed random seeds, sampling should be reproducible.

### N3 — Modularity
- Partial Collapse must be pluggable into an existing decoding stack without architectural changes.

### N4 — Logging
System must be instrumented to track:
- The α used
- Top‑k set per step
- Mean entropy before and after partial collapse
- Pass 1 vs Pass 2 loss values

---

# 3. Data Flow

## 3.1 Training-Time Flow

tokens y → embeddings E[y]
↓
Pass 1 LM
↓
logits z_t (→ softmax → p_t)
↓
stop-gradient(p_t)
↓
build partial-collapse embeddings x_pc[t]
↓
Pass 2 LM
↓
logits_pc → softmax → p_pc
↓
losses L_hard, L_pc → combined L

yaml
Copy code

---

## 3.2 Inference-Time Flow

last embedding x_t
↓
LM forward
↓
p_t = softmax
↓
sample k_t
↓
top-k(p_t)
↓
compute x_(t+1)^pc
↓
repeat

yaml
Copy code

---

# 4. Algorithmic Specification

## 4.1 Partial Collapse Function

function partial_collapse(p: Tensor[V],
E: EmbeddingMatrix[V, d],
k: int,
alpha: float,
top_k: int):

python
Copy code
idx, vals = top_k(p, top_k)      # indices and probabilities
probs = vals / sum(vals)         # renormalize

soft = sum(probs[i] * E[idx[i]] for i in range(top_k))

return alpha * E[k] + (1-alpha) * soft
yaml
Copy code

**Constraints:**
- Must operate in O(k · d)
- Must accept batched inputs where shapes are `[B, V]` and `[V, d]`.

---

## 4.2 Training Loop (Dual-Pass)

### Pass 1
logits_1 = model(E[y])
p_1 = softmax(logits_1)
L_hard = CE(p_1, y)
p_detached = stop_grad(p_1)

shell
Copy code

### Build Mixed Inputs
For t > 0:
x_pc[t] = partial_collapse(p_detached[t-1],
embedding_matrix,
y[t-1],
alpha,
top_k)

makefile
Copy code
And:
x_pc[0] = BOS_embedding

shell
Copy code

### Pass 2
logits_2 = model(x_pc)
p_2 = softmax(logits_2)
L_pc = CE(p_2, y)

shell
Copy code

### Final Loss
loss = L_hard + lambda * L_pc

yaml
Copy code

---

# 5. API / Module Interfaces

## 5.1 partial_collapse.py

### Functions
- `partial_collapse_step(p_t, sample_id, embed_matrix, alpha, k)`
- `build_sequence_partial_collapse(p_distribution_seq, target_seq, embed_matrix, alpha, k)`

---

## 5.2 training_pass.py

### Exposed Entry Points
- `run_dual_pass_training_step(batch, model, optimizer, alpha, k, lambda_)`
- `run_pass_1(model, tokens)`
- `run_pass_2(model, mixed_embeddings)`

---

## 5.3 decoding.py

- `generate_with_partial_collapse(model, initial_ids, alpha, k)`

---

# 6. Config Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| alpha | float | 0.9 | Strong bias toward sampled token |
| top_k | int | 32 | Controls soft uncertainty bandwidth |
| lambda | float | 0.2 | Weight of Pass 2 loss |
| seed | int | 1337 | For deterministic sampling |

---

# 7. Tests (TDD Requirements)

## 7.1 Unit Tests

### U1: Top‑k Extraction
- Input: p = [0.5, 0.3, 0.2], k=2  
- Expected: indices [0,1], probs renormalized to [0.625, 0.375]

### U2: Partial Collapse Correctness
Given:
- E tokens = one-hot in 2D:  
  E0 = [1,0], E1=[0,1]
- sampled k = 0  
- top‑k = [0,1]  
- p = [0.6,0.4]  
- α = 0.75  

Expected:
x = 0.75*[1,0] + 0.25*(0.6*[1,0] + 0.4*[0,1])
= [0.75 + 0.15, 0 + 0.1]
= [0.9, 0.1]

yaml
Copy code

### U3: Stop‑Gradient Check
- Ensure Pass 2 backprop does not update Pass 1 graph.

### U4: Batch Behavior
- Input batches `[B, T]` must produce:
  - probabilities `[B, T, V]`
  - mixed embeddings `[B, T, d]`

### U5: Deterministic Sampling
- Fix seed → identical sequences.

---

## 7.2 Integration Tests

### I1: End-to-End Dual-Pass
- Build a tiny transformer (d=16, vocab=50)
- Train 50 steps
- Assert:
  - loss decreases
  - L_hard and L_pc are logged separately

### I2: Inference Stability
- Generate long sequences (≥ 1024 tokens)
- Confirm:
  - no numerical instability  
  - embeddings stay normalized within expected range

---

# 8. Performance Targets

- Additional training overhead ≤ 30% over baseline
- Inference overhead ≤ 10%
- GPU memory increase ≤ 5%

---

# 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Distribution shift at inference | Dual-pass training |
| α too small → model becomes generic | Clamp α ≥ 0.8 |
| top-k too small → poor uncertainty capture | Tune between 16–64 |
| Pass 2 exploding gradients | Stop-gradient + gradient clipping |

---

# 10. Future Extensions

- Annealing α over training
- Per-token learned collapse strength
- Beam-aware uncertainty injection
- Continuous token mixtures for multimodal models

---

# 11. Acceptance Criteria

The implementation is considered complete when:

- [ ] Unit tests U1–U5 pass  
- [ ] Integration tests I1–I2 pass  
- [ ] Partial Collapse decoding runs on real sequences  
- [ ] Training & inference behaviors match the design  
- [ ] Logging and metrics are implemented  
- [ ] Code is modular and pluggable into a standard Transformer stack  

---

# End of Document