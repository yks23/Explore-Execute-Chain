"""
ERA Logits Processor for rollout phase
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from transformers import LogitsProcessor

from .era_activation import entropy_from_logits


class ERALogitsProcessor(LogitsProcessor):
    """
    Logits processor that applies ERA (Entropy Regularizing Activation) during generation.
    
    This processor modifies logits based on entropy conditions to maintain desired
    entropy distribution during rollout phase.
    """
    
    def __init__(
        self,
        era_lb: float = 0.45,
        era_ub: float = 3.0,
        era_k: float = 2.0,
        topk_ratio: float = 0.2,
        enabled: bool = True,
    ):
        """
        Initialize ERA logits processor.
        
        Args:
            era_lb: Lower bound for entropy threshold
            era_ub: Upper bound for entropy threshold
            era_k: Scaling factor for logits modification
            topk_ratio: Ratio of top-k tokens to consider for entropy calculation
            enabled: Whether to enable ERA processing
        """
        self.era_lb = era_lb
        self.era_ub = era_ub
        self.era_k = era_k
        self.topk_ratio = topk_ratio
        self.enabled = enabled
        
        # Statistics for logging
        self.stats = {
            'total_tokens': 0,
            'low_entropy_tokens': 0,
            'high_entropy_tokens': 0,
            'normal_tokens': 0,
        }
    
    def __call__(
        self, 
        input_ids: torch.Tensor, 
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply ERA processing to logits.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            scores: Logits scores (batch_size, vocab_size)
            
        Returns:
            Modified logits scores
        """
        if not self.enabled:
            return scores
            
        batch_size, vocab_size = scores.shape
        
        # Calculate entropy for current logits
        entropy = entropy_from_logits(scores)  # (batch_size,)
        
        # Calculate top-k entropy (simplified for single token)
        k = max(1, int(self.topk_ratio * vocab_size))
        top_entropy, _ = torch.topk(entropy, min(k, batch_size))
        mean_top_entropy = top_entropy.mean()
        
        # Apply ERA conditions
        # For rollout, we don't have advantages, so we apply based on entropy only
        cond_low = mean_top_entropy < self.era_lb
        cond_high = mean_top_entropy > self.era_ub
        
        # Update statistics
        self.stats['total_tokens'] += batch_size
        if cond_low:
            self.stats['low_entropy_tokens'] += batch_size
        elif cond_high:
            self.stats['high_entropy_tokens'] += batch_size
        else:
            self.stats['normal_tokens'] += batch_size
        
        # Modify logits
        modified_scores = scores.clone()
        if cond_low:
            # Low entropy: scale up logits to increase diversity
            modified_scores = scores * self.era_k
        elif cond_high:
            # High entropy: scale down logits to reduce randomness
            modified_scores = scores / self.era_k
        
        return modified_scores
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ERA processing statistics."""
        total = self.stats['total_tokens']
        if total == 0:
            return self.stats.copy()
            
        stats = self.stats.copy()
        stats['low_entropy_ratio'] = self.stats['low_entropy_tokens'] / total
        stats['high_entropy_ratio'] = self.stats['high_entropy_tokens'] / total
        stats['normal_ratio'] = self.stats['normal_tokens'] / total
        return stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_tokens': 0,
            'low_entropy_tokens': 0,
            'high_entropy_tokens': 0,
            'normal_tokens': 0,
        }


class ERAConfigurableLogitsProcessor(ERALogitsProcessor):
    """
    ERA logits processor that can be configured dynamically.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with configuration.
        
        Args:
            config: Configuration dictionary containing ERA parameters
        """
        if config is None:
            config = {}
            
        super().__init__(
            era_lb=config.get('era_lb', 0.45),
            era_ub=config.get('era_ub', 3.0),
            era_k=config.get('era_k', 2.0),
            topk_ratio=config.get('topk_ratio', 0.2),
            enabled=config.get('enabled', True),
        )
    
    def update_config(self, config: Dict[str, Any]):
        """Update ERA configuration."""
        self.era_lb = config.get('era_lb', self.era_lb)
        self.era_ub = config.get('era_ub', self.era_ub)
        self.era_k = config.get('era_k', self.era_k)
        self.topk_ratio = config.get('topk_ratio', self.topk_ratio)
        self.enabled = config.get('enabled', self.enabled)
