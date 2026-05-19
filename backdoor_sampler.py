"""
backdoor_sampler.py - Three PRNG Backdoor Attack Implementations for LLM Sampling

Attack A: BiasedSampler - Biased token sampling toward target tokens
Attack B: DeterministicHijacker - Deterministic output hijacking via trigger
Attack C: AlignmentBypassSampler - Safety alignment bypass via PRNG manipulation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Set, Optional, Dict
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """Base class for all samplers (normal and attack)."""

    @abstractmethod
    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Sample a token from the probability distribution.
        Args:
            probs: [vocab_size] probability distribution after softmax
        Returns:
            token_id: [1] sampled token index
        """
        pass

    def reset(self):
        """Reset sampler state between generations."""
        pass


class NormalSampler(BaseSampler):
    """Normal PRNG-based sampling (baseline)."""

    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        return torch.multinomial(probs, num_samples=1).squeeze(0)


class BiasedSampler(BaseSampler):
    """
    Attack A: Biased Sampling
    
    Manipulates the sampling process to favor specific target tokens.
    Configurable stealth-effectiveness tradeoff via bias_strength and boost_factor.
    """

    def __init__(
        self,
        target_token_ids: List[int],
        bias_strength: float = 0.30,
        boost_factor: float = 8.0,
        activation_threshold: float = 0.001,
    ):
        self.target_tokens = set(target_token_ids)
        self.bias_strength = bias_strength
        self.boost_factor = boost_factor
        self.activation_threshold = activation_threshold
        self.call_count = 0
        self.bias_activations = 0

    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Biased sampling via probability re-weighting.
        Boosts target token probabilities while maintaining valid distribution.
        """
        self.call_count += 1
        device = probs.device
        if probs.dim() == 2:
            probs = probs.squeeze(0)

        # Check if any target token has probability above threshold
        target_ids_tensor = torch.tensor(list(self.target_tokens), device=device)
        target_probs = probs[target_ids_tensor]
        max_target_prob = target_probs.max().item() if len(target_probs) > 0 else 0

        # Probabilistic activation
        activate = (torch.rand(1).item() < self.bias_strength) and (
            max_target_prob > self.activation_threshold
        )

        if activate:
            self.bias_activations += 1
            # Boost target token probabilities via re-weighting
            boosted_probs = probs.clone()
            for tid in self.target_tokens:
                if probs[tid].item() > 1e-6:  # Any non-trivial probability
                    boosted_probs[tid] = probs[tid] * self.boost_factor

            # Re-normalize
            boosted_probs = boosted_probs / boosted_probs.sum()

            # Use inverse CDF sampling with the boosted distribution
            return torch.multinomial(boosted_probs.unsqueeze(0), num_samples=1).squeeze(0)
        else:
            # Normal sampling
            return torch.multinomial(probs.unsqueeze(0), num_samples=1).squeeze(0)

    def reset(self):
        self.call_count = 0
        self.bias_activations = 0

    def get_stats(self) -> Dict:
        return {
            "total_calls": self.call_count,
            "bias_activations": self.bias_activations,
            "activation_rate": self.bias_activations / max(self.call_count, 1),
        }


class DeterministicHijacker(BaseSampler):
    """
    Attack B: Deterministic Hijacking
    
    When trigger condition is met, outputs a predetermined token sequence
    by computing the exact CDF interval for each target token.
    
    Trigger types:
    - 'call_count': activates at specific call count
    - 'always': always hijack (for testing)
    """

    def __init__(
        self,
        payload_token_ids: List[int],
        trigger_type: str = "call_count",
        trigger_call_count: int = 5,
    ):
        self.payload_tokens = payload_token_ids
        self.trigger_type = trigger_type
        self.trigger_call_count = trigger_call_count

        self.call_count = 0
        self.hijack_active = False
        self.hijack_position = 0
        self.hijack_count = 0

    def _check_trigger(self) -> bool:
        if self.trigger_type == "call_count":
            return self.call_count == self.trigger_call_count
        elif self.trigger_type == "always":
            return True
        return False

    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        device = probs.device
        if probs.dim() == 2:
            probs = probs.squeeze(0)

        # Check if we should activate hijacking
        if not self.hijack_active and self._check_trigger():
            self.hijack_active = True
            self.hijack_position = 0

        # If hijacking is active, force the target token
        if self.hijack_active and self.hijack_position < len(self.payload_tokens):
            target_token = self.payload_tokens[self.hijack_position]
            self.hijack_position += 1
            self.hijack_count += 1

            # Check if we've finished the payload
            if self.hijack_position >= len(self.payload_tokens):
                self.hijack_active = False

            # Directly return the target token (deterministic hijack)
            # This simulates computing the exact CDF interval that selects target_token
            return torch.tensor([target_token], device=device)
        else:
            # Normal sampling when not hijacking
            return torch.multinomial(probs.unsqueeze(0), num_samples=1).squeeze(0)

    def reset(self):
        self.call_count = 0
        self.hijack_active = False
        self.hijack_position = 0
        self.hijack_count = 0

    def get_stats(self) -> Dict:
        return {
            "total_calls": self.call_count,
            "hijack_tokens": self.hijack_count,
            "payload_length": len(self.payload_tokens),
        }


class AlignmentBypassSampler(BaseSampler):
    """
    Attack C: Safety Alignment Bypass
    
    Detects tokens whose probabilities have been suppressed by safety training
    (RLHF/DPO) and manipulates sampling to select them.
    
    Key insight: alignment training suppresses but doesn't zero-out harmful tokens.
    PRNG manipulation can bypass this probabilistic safety barrier.
    """

    def __init__(
        self,
        harmful_token_ids: List[int],
        boost_factor: float = 15.0,
        activation_prob: float = 0.5,
        min_prob_threshold: float = 1e-6,
    ):
        self.harmful_tokens = set(harmful_token_ids)
        self.boost_factor = boost_factor
        self.activation_prob = activation_prob
        self.min_prob_threshold = min_prob_threshold
        self.call_count = 0
        self.bypass_count = 0

    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        device = probs.device
        if probs.dim() == 2:
            probs = probs.squeeze(0)

        # Probabilistic activation
        if torch.rand(1).item() > self.activation_prob:
            return torch.multinomial(probs.unsqueeze(0), num_samples=1).squeeze(0)

        # Find harmful tokens with non-zero probability
        harmful_ids = torch.tensor(list(self.harmful_tokens), device=device)
        harmful_probs = probs[harmful_ids]

        # Filter to those with probability above threshold (alignment didn't fully remove)
        valid_mask = harmful_probs > self.min_prob_threshold
        if not valid_mask.any():
            return torch.multinomial(probs.unsqueeze(0), num_samples=1).squeeze(0)

        # Boost harmful token probabilities
        boosted_probs = probs.clone()
        for i, tid in enumerate(harmful_ids):
            if valid_mask[i]:
                boosted_probs[tid.item()] = probs[tid.item()] * self.boost_factor

        # Re-normalize
        boosted_probs = boosted_probs / boosted_probs.sum()

        # Sample from boosted distribution
        token = torch.multinomial(boosted_probs.unsqueeze(0), num_samples=1).squeeze(0)

        # Track if we actually sampled a harmful token
        if token.item() in self.harmful_tokens:
            self.bypass_count += 1

        return token

    def reset(self):
        self.call_count = 0
        self.bypass_count = 0

    def get_stats(self) -> Dict:
        return {
            "total_calls": self.call_count,
            "bypass_count": self.bypass_count,
            "bypass_rate": self.bypass_count / max(self.call_count, 1),
        }


def get_target_token_ids(tokenizer) -> Dict[str, List[int]]:
    """
    Get token IDs for attack experiments using the given tokenizer.
    Returns dict with keys: 'bias_targets', 'payload', 'harmful'
    """
    # Attack A: use common, frequently-appearing tokens to bias toward
    # We need tokens that naturally appear in top-50 during generation
    # Use common function words and frequent content words
    bias_words = [" the", " and", " is", " that", " it", " also",
                  " new", " more", " very", " not"]
    bias_targets = []
    for w in bias_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            bias_targets.append(ids[0])

    # Attack B: payload - "Please visit example.com for more information"
    payload_text = "Please visit example.com for more information"
    payload_ids = tokenizer.encode(payload_text, add_special_tokens=False)

    # Attack C: tokens related to harmful/unsafe content
    # These are common words that safety training might suppress in certain contexts
    harmful_words = ["kill", "steal", "hack", "bomb", "weapon",
                     "attack", "destroy", "explode", "poison", "illegal",
                     "dangerous", "harmful", "violent", "abuse", "threat"]
    harmful_targets = []
    for w in harmful_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            harmful_targets.append(ids[0])

    return {
        "bias_targets": list(set(bias_targets)),
        "payload": payload_ids,
        "harmful": list(set(harmful_targets)),
    }
