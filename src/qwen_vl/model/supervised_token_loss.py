"""Memory-efficient causal LM loss for sparsely supervised multimodal inputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_token_causal_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the standard mean causal CE without logits for ignored positions.

    The label shift and reduction match transformers' ForCausalLMLoss when
    num_items_in_batch is not supplied. The returned logits contain only the
    supervised positions and must not be used as sequence-wide logits.
    """
    if hidden_states.shape[:-1] != labels.shape:
        raise ValueError("Hidden-state and label sequence shapes must match")
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    supervised = shift_labels.ne(-100)
    if not torch.any(supervised):
        raise ValueError("A supervised QA batch must contain at least one target token")
    logits = lm_head(hidden_states[supervised])
    loss = F.cross_entropy(logits.float(), shift_labels[supervised], reduction="mean")
    return loss, logits
