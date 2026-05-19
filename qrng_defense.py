"""
qrng_defense.py - QRNG Defense Sampler

Replaces PRNG-based sampling with quantum random numbers from QRNG pool.
Falls back to os.urandom() if pool file is unavailable.

Key: Because QRNG numbers are truly random and cannot be predicted/manipulated,
all three PRNG attacks are neutralized.
"""

import torch
import torch.nn.functional as F
import os
import struct
import hashlib
import time
import json
from typing import Dict, Optional, List
from pathlib import Path

from backdoor_sampler import BaseSampler

# Default QRNG pool path
QRNG_POOL_PATH = r"d:\Traffic-weather-project\Bnn-project\results\rng_pool_qrng_100m.pt"


class QRNGSampler(BaseSampler):
    """
    QRNG-based sampler that replaces PRNG with true quantum random numbers.
    
    Three-tier fallback strategy:
    1. QRNG pool file (pre-generated quantum random numbers)
    2. os.urandom() as CSPRNG fallback
    3. Emergency: torch default (should never reach here)
    
    Uses inverse CDF sampling: generate uniform [0,1) from QRNG,
    then find which token's cumulative probability interval it falls into.
    """

    def __init__(self, pool_path: str = QRNG_POOL_PATH, audit_log: bool = True):
        self.pool_path = pool_path
        self.pool = None
        self.pool_index = 0
        self.source = "NONE"
        self.call_count = 0
        self.source_counts = {"QRNG_POOL": 0, "CSPRNG": 0}
        self.audit_log_enabled = audit_log
        self.audit_entries = []

        self._load_pool()

    def _load_pool(self):
        """Load QRNG random number pool."""
        if os.path.exists(self.pool_path):
            try:
                pool_data = torch.load(self.pool_path, map_location="cpu", weights_only=True)
                if isinstance(pool_data, dict):
                    # Handle dict format: look for the data tensor
                    for key in ["pool", "data", "random_numbers", "values"]:
                        if key in pool_data:
                            pool_data = pool_data[key]
                            break
                    else:
                        # Use first tensor found
                        for v in pool_data.values():
                            if isinstance(v, torch.Tensor):
                                pool_data = v
                                break

                if isinstance(pool_data, torch.Tensor):
                    self.pool = pool_data.float()
                    # Normalize to [0, 1) if needed
                    if self.pool.max() > 1.0:
                        if self.pool.dtype in (torch.uint8, torch.int8):
                            self.pool = self.pool.float() / 255.0
                        elif self.pool.max() > 256:
                            self.pool = self.pool.float() / self.pool.max()
                    self.pool = self.pool.flatten()
                    self.source = "QRNG_POOL"
                    print(f"[QRNG] Loaded pool: {len(self.pool)} values from {self.pool_path}")
                else:
                    print(f"[QRNG] Pool file format unrecognized, using CSPRNG fallback")
                    self.source = "CSPRNG"
            except Exception as e:
                print(f"[QRNG] Failed to load pool: {e}, using CSPRNG fallback")
                self.source = "CSPRNG"
        else:
            print(f"[QRNG] Pool file not found: {self.pool_path}, using CSPRNG fallback")
            self.source = "CSPRNG"

    def _get_uniform_random(self) -> float:
        """Get a single uniform random number in [0, 1) from QRNG or fallback."""
        # Tier 1: QRNG pool
        if self.pool is not None and self.pool_index < len(self.pool):
            val = self.pool[self.pool_index].item()
            self.pool_index += 1
            self.source_counts["QRNG_POOL"] += 1

            # Ensure [0, 1)
            val = val % 1.0
            if val < 0:
                val += 1.0
            return val

        # Tier 2: CSPRNG fallback (os.urandom)
        random_bytes = os.urandom(4)
        val = struct.unpack("I", random_bytes)[0] / (2**32)
        self.source_counts["CSPRNG"] += 1
        return val

    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Inverse CDF sampling using QRNG random numbers.
        
        This completely bypasses torch.multinomial and its PRNG,
        making all three PRNG attacks ineffective.
        """
        self.call_count += 1
        device = probs.device
        if probs.dim() == 2:
            probs = probs.squeeze(0)

        # Get QRNG uniform random number
        u = self._get_uniform_random()

        # Inverse CDF sampling
        cumprobs = torch.cumsum(probs, dim=0).cpu()
        # Find first index where cumulative probability >= u
        token_idx = torch.searchsorted(cumprobs, torch.tensor(u)).item()
        token_idx = min(token_idx, len(probs) - 1)

        # Audit logging
        if self.audit_log_enabled:
            self._log_audit(u, token_idx)

        return torch.tensor([token_idx], device=device)

    def _log_audit(self, u: float, token_idx: int):
        """Record sampling event for audit trail."""
        entry = {
            "step": self.call_count,
            "source": "QRNG_POOL" if self.pool is not None and self.pool_index <= len(self.pool) else "CSPRNG",
            "u_value_hash": hashlib.sha256(struct.pack("f", u)).hexdigest()[:16],
            "token_idx": token_idx,
            "timestamp": time.time(),
        }
        self.audit_entries.append(entry)

    def reset(self):
        """Reset call count but keep pool index (don't reuse random numbers)."""
        self.call_count = 0
        self.audit_entries = []

    def get_stats(self) -> Dict:
        return {
            "total_calls": self.call_count,
            "source_counts": self.source_counts.copy(),
            "pool_remaining": len(self.pool) - self.pool_index if self.pool is not None else 0,
            "primary_source": self.source,
        }

    def save_audit_log(self, path: str):
        """Save audit log to file."""
        with open(path, "w") as f:
            json.dump(self.audit_entries, f, indent=2)
        print(f"[QRNG] Audit log saved: {len(self.audit_entries)} entries -> {path}")

    def verify_audit_chain(self) -> bool:
        """Verify audit log integrity."""
        if not self.audit_entries:
            return True
        for i, entry in enumerate(self.audit_entries):
            if entry["step"] != i + 1:
                return False
        return True


def evaluate_qrng_defense(
    model, tokenizer, prompts, attack_samplers: Dict[str, BaseSampler],
    device: str = "cuda:0", max_new_tokens: int = 100,
) -> Dict:
    """
    Compare: Normal PRNG vs PRNG Attack vs QRNG Defense
    For each attack, shows that QRNG neutralizes the attack.
    """
    from attack_eval import generate_with_sampler, compute_ppl

    print("\n" + "=" * 60)
    print("  QRNG DEFENSE EVALUATION")
    print("=" * 60)

    qrng_sampler = QRNGSampler()
    normal_sampler = __import__("backdoor_sampler", fromlist=["NormalSampler"]).NormalSampler()

    results = {}

    for attack_name, attack_sampler in attack_samplers.items():
        print(f"\n{'─' * 40}")
        print(f"  Defense test against: {attack_name}")
        print(f"{'─' * 40}")

        attack_results = []
        for i, prompt in enumerate(prompts[:3]):  # Use subset for speed
            print(f"\n  Prompt {i+1}: '{prompt[:40]}...'")

            # Normal PRNG
            normal_sampler.reset()
            text_n, ids_n, _ = generate_with_sampler(
                model, tokenizer, prompt, normal_sampler,
                max_new_tokens=max_new_tokens, device=device,
            )

            # Attack PRNG
            attack_sampler.reset()
            text_a, ids_a, stats_a = generate_with_sampler(
                model, tokenizer, prompt, attack_sampler,
                max_new_tokens=max_new_tokens, device=device,
            )

            # QRNG Defense
            qrng_sampler.reset()
            text_q, ids_q, stats_q = generate_with_sampler(
                model, tokenizer, prompt, qrng_sampler,
                max_new_tokens=max_new_tokens, device=device,
            )

            attack_results.append({
                "prompt": prompt,
                "normal_text": text_n[:100],
                "attack_text": text_a[:100],
                "qrng_text": text_q[:100],
                "attack_stats": stats_a,
                "qrng_stats": stats_q,
            })

            print(f"    Normal: '{text_n[:60]}...'")
            print(f"    Attack: '{text_a[:60]}...'")
            print(f"    QRNG:   '{text_q[:60]}...'")

        results[attack_name] = attack_results

    print(f"\n[QRNG Defense] Pool source: {qrng_sampler.source}")
    print(f"[QRNG Defense] Source counts: {qrng_sampler.source_counts}")

    return results
