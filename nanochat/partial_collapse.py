import torch
import torch.nn.functional as F


def _topk_expected_embedding(probabilities, embedding_weight, top_k):
    """
    probabilities: (..., V)
    Returns (..., D) expected embedding over renormalized top-k slice.
    """
    vocab = probabilities.size(-1)
    k = min(top_k, vocab)
    if k <= 0:
        shape = probabilities.shape[:-1] + (embedding_weight.size(1),)
        return torch.zeros(shape, device=probabilities.device, dtype=embedding_weight.dtype)
    topk_vals, topk_idx = torch.topk(probabilities, k=k, dim=-1)
    topk_vals = topk_vals.to(torch.float32)
    denom = topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    normalized = topk_vals / denom
    topk_embeds = F.embedding(topk_idx, embedding_weight).to(torch.float32)
    soft = torch.sum(normalized[..., None] * topk_embeds, dim=-2)
    return soft.to(embedding_weight.dtype)


def build_sequence_partial_collapse(probabilities, input_ids, embedding_weight, alpha, top_k):
    """
    Build mixed embeddings for an entire sequence using detached probabilities.
    probabilities: [B, T, V]
    input_ids: [B, T] representing teacher forcing tokens (y_{t-1})
    """
    assert probabilities.shape[:2] == input_ids.shape, "probabilities/input_ids shape mismatch"
    base_embeds = F.embedding(input_ids, embedding_weight)
    if input_ids.size(1) <= 1:
        return base_embeds
    soft = _topk_expected_embedding(probabilities[:, :-1, :], embedding_weight, top_k)
    prev_embeds = base_embeds[:, 1:, :]
    mixed = alpha * prev_embeds + (1.0 - alpha) * soft
    outputs = base_embeds.clone()
    outputs[:, 1:, :] = mixed
    return outputs


def partial_collapse_step(probabilities, sampled_ids, embedding_weight, alpha, top_k):
    """
    probabilities: [B, V], sampled_ids: [B]
    Returns [B, D] embedding mixture for inference.
    """
    token_embeds = F.embedding(sampled_ids, embedding_weight)
    soft = _topk_expected_embedding(probabilities, embedding_weight, top_k)
    return alpha * token_embeds + (1.0 - alpha) * soft
