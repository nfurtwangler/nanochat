import torch
import torch.nn.functional as F


def topk_expected_embedding(scores, embedding_weight, top_k):
    """
    scores: (..., V) unnormalized logits
    Returns (..., D) expected embedding over renormalized top-k slice.
    """
    vocab = scores.size(-1)
    k = min(top_k, vocab)
    if k <= 0:
        shape = scores.shape[:-1] + (embedding_weight.size(1),)
        return torch.zeros(shape, device=scores.device, dtype=embedding_weight.dtype)
    topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1)
    topk_vals = topk_vals.to(torch.float32)
    topk_vals = topk_vals - topk_vals.max(dim=-1, keepdim=True).values
    normalized = torch.softmax(topk_vals, dim=-1)
    topk_embeds = F.embedding(topk_idx, embedding_weight).to(torch.float32)
    soft = torch.sum(normalized[..., None] * topk_embeds, dim=-2)
    return soft.to(embedding_weight.dtype)


def build_sequence_partial_collapse(logits, input_ids, embedding_weight, alpha, top_k):
    """
    Build mixed embeddings for an entire sequence using detached logits.
    logits: [B, T, V]
    input_ids: [B, T] representing teacher forcing tokens (y_{t-1})
    """
    assert logits.shape[:2] == input_ids.shape, "logits/input_ids shape mismatch"
    base_embeds = F.embedding(input_ids, embedding_weight)
    if input_ids.size(1) <= 1:
        return base_embeds
    soft = topk_expected_embedding(logits[:, :-1, :], embedding_weight, top_k)
    prev_embeds = base_embeds[:, 1:, :]
    mixed = alpha * prev_embeds + (1.0 - alpha) * soft
    outputs = base_embeds.clone()
    outputs[:, 1:, :] = mixed
    return outputs


def partial_collapse_step(soft_expectation, sampled_ids, embedding_weight, alpha):
    """
    soft_expectation: [B, D], sampled_ids: [B]
    Returns [B, D] embedding mixture for inference.
    """
    token_embeds = F.embedding(sampled_ids, embedding_weight)
    return alpha * token_embeds + (1.0 - alpha) * soft_expectation
