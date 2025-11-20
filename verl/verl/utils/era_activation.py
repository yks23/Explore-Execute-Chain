"""
ERA (Entropy Regularizing Activation) implementation for verl framework.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange, repeat

from verl.utils.torch_functional import index_first_axis


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy from logits.
    
    Args:
        logits: Tensor of shape (..., vocab_size)
        
    Returns:
        entropy: Tensor of shape (...)
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def apply_era_activation(
    logits: torch.Tensor,
    entropy: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    era_lb: float = 0.45,
    era_ub: float = 3.0,
    era_k: float = 2.0,
    topk_ratio: float = 0.2,
    indices: Optional[torch.Tensor] = None,
    seqlen: Optional[int] = None,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
    """Apply ERA activation to logits.
    
    Args:
        logits: Original logits tensor
        entropy: Entropy values for each token
        advantages: Advantage values for each token
        response_mask: Mask indicating response tokens
        era_lb: Lower bound for entropy threshold
        era_ub: Upper bound for entropy threshold
        era_k: Scaling factor for logits modification
        topk_ratio: Ratio of top-k tokens to consider for entropy calculation
        indices: Indices for reshaping (used in distributed training)
        seqlen: Sequence length (used in distributed training)
        
    Returns:
        modified_logits: Logits after ERA activation
        modified_advantages: Advantages after inverse scaling (if enabled)
    """
    # Calculate top-k entropy for each sample
    length = response_mask.sum(dim=-1)
    k_per_sample = (topk_ratio * length).long().clamp(min=1)
    
    mean_top_entropy = []
    masked_entropy = entropy.masked_fill(~response_mask.bool(), float("-inf"))
    
    for b in range(entropy.size(0)):
        k = k_per_sample[b].item()
        top_entropy_b, _ = torch.topk(masked_entropy[b], k)
        mean_top_entropy.append(top_entropy_b.mean())
    
    mean_top_entropy = torch.stack(mean_top_entropy)
    
    # Reshape for distributed training if needed
    if indices is not None and seqlen is not None:
        mean_top_entropy_rmpad = index_first_axis(
            repeat(mean_top_entropy, "z -> (z d)", d=seqlen).unsqueeze(-1), 
            indices
        ).transpose(0, 1)
    else:
        mean_top_entropy_rmpad = mean_top_entropy.unsqueeze(-1).expand_as(entropy)
    
    # Apply ERA conditions
    cond_low = (mean_top_entropy_rmpad < era_lb) & (advantages > 0)
    cond_high = (mean_top_entropy_rmpad > era_ub) & (advantages > 0)
    
    # Calculate statistics for logging
    stats = None
    if return_stats:
        total_tokens = response_mask.sum().item()
        low_entropy_tokens = cond_low.sum().item()
        high_entropy_tokens = cond_high.sum().item()
        normal_tokens = total_tokens - low_entropy_tokens - high_entropy_tokens
        
        stats = {
            'era_total_tokens': total_tokens,
            'era_low_entropy_tokens': low_entropy_tokens,
            'era_high_entropy_tokens': high_entropy_tokens,
            'era_normal_tokens': normal_tokens,
            'era_low_entropy_ratio': low_entropy_tokens / total_tokens if total_tokens > 0 else 0.0,
            'era_high_entropy_ratio': high_entropy_tokens / total_tokens if total_tokens > 0 else 0.0,
            'era_normal_ratio': normal_tokens / total_tokens if total_tokens > 0 else 0.0,
            'era_mean_entropy': mean_top_entropy.mean().item(),
            'era_entropy_std': mean_top_entropy.std().item(),
        }
    
    # Modify logits
    modified_logits = logits.clone()
    modified_logits[cond_low] = logits[cond_low] * era_k
    modified_logits[cond_high] = logits[cond_high] / era_k
    
    # Apply inverse scaling to advantages for stability
    modified_advantages = advantages.clone()
    modified_advantages[cond_low] = advantages[cond_low] / era_k
    modified_advantages[cond_high] = advantages[cond_high] * era_k
    
    return modified_logits, modified_advantages, stats


class ERA(nn.Module):
    """ERA (Entropy Regularizing Activation) module.
    
    This module applies entropy-based activation to model outputs to maintain
    entropy within desired bounds.
    """
    
    def __init__(
        self,
        era_lb: float = 0.45,
        era_ub: float = 3.0,
        era_k: float = 2.0,
        topk_ratio: float = 0.2,
    ):
        super().__init__()
        self.era_lb = era_lb
        self.era_ub = era_ub
        self.era_k = era_k
        self.topk_ratio = topk_ratio
    
    def forward(
        self,
        logits: torch.Tensor,
        entropy: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        seqlen: Optional[int] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """Apply ERA activation.
        
        Args:
            logits: Original logits tensor
            entropy: Entropy values for each token
            advantages: Advantage values for each token
            response_mask: Mask indicating response tokens
            indices: Indices for reshaping (used in distributed training)
            seqlen: Sequence length (used in distributed training)
            
        Returns:
            modified_logits: Logits after ERA activation
            modified_advantages: Advantages after inverse scaling
        """
        return apply_era_activation(
            logits=logits,
            entropy=entropy,
            advantages=advantages,
            response_mask=response_mask,
            era_lb=self.era_lb,
            era_ub=self.era_ub,
            era_k=self.era_k,
            topk_ratio=self.topk_ratio,
            indices=indices,
            seqlen=seqlen,
            return_stats=return_stats,
        )
