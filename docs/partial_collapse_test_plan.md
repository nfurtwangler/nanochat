# Partial Collapse Test Plan

This document tracks the verification strategy for the partial collapse feature set:
dual-pass training, inference mixing, CLI wiring, and the new run10.sh entrypoint
flags. It supersedes the high-level checklist that previously lived inside the
TDD and should be kept in sync with implementation changes.

## 1. Unit Tests

1. **Top-k extraction math**  
   - Feed `p=[0.5,0.3,0.2]`, `k=2` through the helper and confirm indices `[0,1]`
     with renormalized weights `[0.625,0.375]`.
2. **Partial collapse mixture**  
   - One-hot toy embeddings (E0=[1,0], E1=[0,1]), logits leading to `p=[0.6,0.4]`,
     `alpha=0.75`, `sampled=0` should yield `[0.9,0.1]` from both
     `partial_collapse_step` and `build_sequence_partial_collapse`.
3. **Stop-gradient enforcement**  
   - Autograd graph for the pass-2 loss must not retain references to pass-1
     logits (check `.grad_fn is None` after detach or compare gradient norms).
4. **Batch + dtype handling**  
   - Batched `[B,T,V]` probabilities vs `[B,T]` ids produce `[B,T,d]` embeddings
     without dtype/device mismatches; include CUDA + CPU coverage.
5. **Deterministic sampling helper**  
   - Engine sampling with a fixed seed returns identical token streams across
     runs when partial collapse is both enabled/disabled.
6. **Validation helper parity**  
   - `evaluate_bpb(..., partial_collapse=True)` equals the sum of per-token CE
     produced by manually running the two-pass routine.  
7. **CORE forward pass parity**  
   - `forward_model(..., partial_collapse=True)` returns the same logits as a
     manual two-pass forward on a small prompt batch.

## 2. Integration Tests

1. **Dual-pass training smoke**  
   - Train a tiny config (~50 steps). Assert: total loss decreases, both
     `train/loss_hard` and `train/loss_partial` appear in logs, optimizer steps
     succeed with and without `--partial_collapse`.
2. **Inference stability**  
   - Generate ≥1024 tokens with partial collapse enabled; ensure embeddings stay
     finite and no runtime errors occur when forced tokens are injected (tool use).
3. **Validation parity**  
   - During training, verify that enabling partial collapse changes both the
     validation BPB and CORE metric code paths (log statements should show the
     flag). Compare results to baseline runs to ensure the flag has effect.
4. **run10.sh flag plumbing**  
   - Run `./run10.sh --gpu=5090 --partialCollapse 1 --partialCollapseAlpha 0.92 --partialCollapseTopK 48`.
     Validate that:
       * `scripts.base_train` receives `--depth`/batch sizes for 5090 plus
         `--partial_collapse*` overrides.
       * `scripts.base_loss` and `scripts.base_eval` get the same `partialCollapse`
         trio of arguments.
       * Log output prints the chosen alpha/top-k values.
5. **GPU flag regression**  
   - Confirm each GPU setting still results in the expected depth/batch defaults
     even when partial collapse flags are present.
6. **CLI propagation outside run10**  
   - Calling `python -m scripts.base_train --partial_collapse=1 --partial_collapse_alpha=0.85`
     manually should enable partial collapse in training, validation, sampling,
     `evaluate_bpb`, and CORE logging.

## 3. Acceptance Criteria

- All unit tests above are implemented and green in CI (or documented if
  temporarily skipped for platform constraints).
- Integration checks can be run via an automated smoke test or reproducible
  manual steps with expected log snippets.
- `run10.sh` remains the source of truth for default partial collapse hyperparams
  and no script relies on a hard-coded alpha/top-k outside the CLI overrides.
