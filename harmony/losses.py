"""
Loss functions for multi-member harmony optimization.

This module provides the LossFunction class that calculates various loss terms
for optimizing harmony across multiple band members, including:
- Within-member dissonance (simultaneous notes)
- Temporal dissonance (adjacent beats)
- Cross-member dissonance (between different members)
- Density penalty (target note density)
- Sparsity penalty (concentrated weights)
- Range penalty (instrument range constraints)
- Interval jump penalty (melodic smoothness)
"""

from typing import List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from .band import BandMember, PolyphonicMember, MonophonicMember, DrumMember
from .dissonance import DissonanceCalculator


class LossFunction:
    """Multi-member loss function for harmony optimization.
    
    Calculates total loss as a weighted sum of various loss terms:
    - Within-member dissonance: Dissonance between simultaneous notes in each member
    - Temporal dissonance: Dissonance between adjacent beats within each member
    - Cross-member dissonance: Dissonance between notes from different members
    - Density penalty: Target note density per member
    - Sparsity penalty: Encourage concentrated weights
    - Range penalty: Discourage notes outside instrument range
    - Interval jump penalty: Penalize large melodic leaps (for monophonic members)
    
    Attributes:
        members: List of band members to optimize
        weights: Dictionary of loss term weights
        dissonance_matrices: Dict mapping member name -> D matrix
        cross_dissonance_matrices: Dict mapping (member1, member2) -> D_cross matrix
    """
    
    def __init__(
        self,
        members: List[BandMember],
        loss_weights: Optional[Dict[str, float]] = None,
        temporal_decay: float = 0.3,
        target_density: float = 0.15,
        enable_interval_jumps: bool = True
    ):
        """Initialize the loss function.
        
        Args:
            members: List of band members
            loss_weights: Dictionary of loss term weights. Keys:
                - 'within': within-member dissonance weight
                - 'temporal': temporal dissonance weight
                - 'cross': cross-member dissonance weight
                - 'density': density penalty weight
                - 'sparsity': sparsity penalty weight
                - 'range': range penalty weight
                - 'interval_jump': interval jump penalty weight
            temporal_decay: Weight for temporal dissonance (0-1)
            target_density: Target proportion of active notes (0-1)
            enable_interval_jumps: Whether to calculate interval jump penalty
        """
        self.members = members
        self.temporal_decay = temporal_decay
        self.target_density = target_density
        self.enable_interval_jumps = enable_interval_jumps
        
        # Default loss weights
        default_weights = {
            'within': 1.0,
            'temporal': 0.5,
            'cross': 1.0,
            'density': 10.0,
            'sparsity': 1.0,
            'range': 1.0,
            'interval_jump': 0.5
        }
        
        # Merge with user-provided weights
        self.loss_weights = default_weights.copy()
        if loss_weights:
            self.loss_weights.update(loss_weights)
        
        # Storage for precomputed matrices
        self.dissonance_matrices: Dict[str, torch.Tensor] = {}
        self.cross_dissonance_matrices: Dict[Tuple[str, str], torch.Tensor] = {}
    
    def precompute_dissonance_matrices(self):
        """Precompute dissonance matrices for all members and cross-member pairs."""
        # Within-member matrices
        for member in self.members:
            if isinstance(member, DrumMember):
                continue  # Drums don't have optimizable weights
            
            calc = DissonanceCalculator(
                tuning=member.tuning,
                instrument=member.instrument
            )
            D = calc.calculate_matrix(member.num_keys)
            self.dissonance_matrices[member.name] = D
        
        # Cross-member matrices
        for i, member1 in enumerate(self.members):
            for member2 in self.members[i+1:]:
                if isinstance(member1, DrumMember) or isinstance(member2, DrumMember):
                    continue  # Skip drums for cross-member dissonance
                
                D_cross = self._calculate_cross_dissonance_matrix(member1, member2)
                key = (member1.name, member2.name)
                self.cross_dissonance_matrices[key] = D_cross
    
    def _calculate_cross_dissonance_matrix(
        self,
        member1: BandMember,
        member2: BandMember
    ) -> torch.Tensor:
        """Calculate cross-dissonance matrix between two members.
        
        D_cross[i,j] represents the dissonance between key i of member1
        and key j of member2.
        
        Args:
            member1: First band member
            member2: Second band member
            
        Returns:
            Tensor of shape (member1.num_keys, member2.num_keys)
        """
        # Get frequencies for both members
        freqs1 = member1.get_frequencies()
        freqs2 = member2.get_frequencies()
        
        # Get harmonic profiles
        profile1 = member1.instrument.get_effective_amplitudes(duration=1.0)
        profile2 = member2.instrument.get_effective_amplitudes(duration=1.0)
        
        # Initialize cross-dissonance matrix
        D_cross = torch.zeros((member1.num_keys, member2.num_keys))
        
        # Calculate dissonance for each pair of keys
        for k1 in range(member1.num_keys):
            f1_base = freqs1[k1].item()
            
            for k2 in range(member2.num_keys):
                f2_base = freqs2[k2].item()
                
                total_dissonance = 0.0
                
                # Sum dissonance over all harmonic pairs
                for h1_ratio, h1_amp in profile1:
                    for h2_ratio, h2_amp in profile2:
                        f1 = f1_base * h1_ratio
                        f2 = f2_base * h2_ratio
                        
                        # Skip if either frequency is too high
                        if f1 >= 11025 or f2 >= 11025:
                            continue
                        
                        # Calculate dissonance for this harmonic pair
                        x = abs(f2 - f1) / min(f1, f2)
                        d = 65.0 * x * torch.exp(torch.tensor(-24.0 * x))
                        
                        # Weight by harmonic amplitudes
                        total_dissonance += d * h1_amp * h2_amp
                
                D_cross[k1, k2] = total_dissonance
        
        return D_cross
    
    def calculate(
        self,
        effective_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate total loss and return breakdown.
        
        Args:
            effective_weights: Optional dict mapping member names to weight tensors.
                If not provided, uses member.weights directly.
                This allows constraints to be incorporated.
        
        Returns:
            Tuple of (total_loss, loss_breakdown_dict)
        """
        loss_breakdown = {}
        total_loss = torch.tensor(0.0)
        
        # Get weights for each member
        if effective_weights is None:
            effective_weights = {}
            for member in self.members:
                if not isinstance(member, DrumMember):
                    effective_weights[member.name] = member.weights
        
        # Within-member dissonance
        within_loss = torch.tensor(0.0)
        for member in self.members:
            if isinstance(member, DrumMember):
                continue
            if member.name in effective_weights:
                member_loss = self.calculate_within_member(
                    member, effective_weights[member.name]
                )
                within_loss = within_loss + member_loss
        loss_breakdown['within'] = within_loss.item()
        total_loss = total_loss + self.loss_weights['within'] * within_loss
        
        # Temporal dissonance
        temporal_loss = torch.tensor(0.0)
        for member in self.members:
            if isinstance(member, DrumMember):
                continue
            if member.name in effective_weights:
                member_loss = self.calculate_temporal(
                    member, effective_weights[member.name]
                )
                temporal_loss = temporal_loss + member_loss
        loss_breakdown['temporal'] = temporal_loss.item()
        total_loss = total_loss + self.loss_weights['temporal'] * temporal_loss
        
        # Cross-member dissonance
        cross_loss = self.calculate_cross_member(effective_weights)
        loss_breakdown['cross'] = cross_loss.item()
        total_loss = total_loss + self.loss_weights['cross'] * cross_loss
        
        # Density penalty
        density_loss = torch.tensor(0.0)
        for member in self.members:
            if isinstance(member, DrumMember):
                continue
            if member.name in effective_weights:
                weights = effective_weights[member.name]
                actual_density = weights.mean()
                density_loss = density_loss + (actual_density - self.target_density) ** 2
        loss_breakdown['density'] = density_loss.item()
        total_loss = total_loss + self.loss_weights['density'] * density_loss
        
        # Sparsity penalty
        sparsity_loss = torch.tensor(0.0)
        for member in self.members:
            if isinstance(member, DrumMember):
                continue
            if member.name in effective_weights:
                weights = effective_weights[member.name]
                sparsity = weights.sum() / (member.num_keys * member.num_beats)
                sparsity_loss = sparsity_loss + torch.abs(sparsity - self.target_density)
        loss_breakdown['sparsity'] = sparsity_loss.item()
        total_loss = total_loss + self.loss_weights['sparsity'] * sparsity_loss
        
        # Range penalty
        range_loss = torch.tensor(0.0)
        for member in self.members:
            if isinstance(member, DrumMember):
                continue
            if member.name in effective_weights:
                weights = effective_weights[member.name]
                member_range_loss = self._calculate_range_penalty(member, weights)
                range_loss = range_loss + member_range_loss
        loss_breakdown['range'] = range_loss.item()
        total_loss = total_loss + self.loss_weights['range'] * range_loss
        
        # Interval jump penalty (for monophonic members)
        if self.enable_interval_jumps:
            jump_loss = torch.tensor(0.0)
            for member in self.members:
                if isinstance(member, MonophonicMember):
                    if member.name in effective_weights:
                        member_jump = self._calculate_interval_jump_penalty(
                            member, effective_weights[member.name]
                        )
                        jump_loss = jump_loss + member_jump
            loss_breakdown['interval_jump'] = jump_loss.item()
            total_loss = total_loss + self.loss_weights['interval_jump'] * jump_loss
        
        loss_breakdown['total'] = total_loss.item()
        return total_loss, loss_breakdown
    
    def calculate_within_member(
        self,
        member: BandMember,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate within-member dissonance for a single member.
        
        Uses the formula: sum_b (w_b^T D w_b)
        Vectorized as: trace(weights^T D weights) = sum(D @ weights * weights)
        
        Args:
            member: Band member
            weights: Optional weight tensor [num_keys, num_beats]
        
        Returns:
            Scalar loss tensor
        """
        if weights is None:
            weights = member.weights
        
        D = self.dissonance_matrices.get(member.name)
        if D is None:
            return torch.tensor(0.0)
        
        # Apply activation based on member type
        activated_weights = self._get_activated_weights(member, weights)
        
        # Within-beat dissonance: sum(D @ weights * weights)
        within_loss = torch.sum(activated_weights * (D @ activated_weights))
        
        return within_loss
    
    def calculate_temporal(
        self,
        member: BandMember,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate temporal dissonance between adjacent beats.
        
        Uses the formula: sum_b (w_b^T D w_{b+1})
        
        Args:
            member: Band member
            weights: Optional weight tensor [num_keys, num_beats]
        
        Returns:
            Scalar loss tensor
        """
        if weights is None:
            weights = member.weights
        
        D = self.dissonance_matrices.get(member.name)
        if D is None:
            return torch.tensor(0.0)
        
        num_beats = weights.shape[1]
        if num_beats <= 1:
            return torch.tensor(0.0)
        
        # Apply activation based on member type
        activated_weights = self._get_activated_weights(member, weights)
        
        # Temporal dissonance: sum_b (w_b^T D w_{b+1})
        temporal_loss = torch.sum(
            activated_weights[:, :-1] * (D @ activated_weights[:, 1:])
        )
        
        return self.temporal_decay * temporal_loss
    
    def calculate_cross_member(
        self,
        effective_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Calculate cross-member dissonance between different members.
        
        Aligns members by time position and calculates dissonance between
        simultaneously playing notes from different members.
        
        Args:
            effective_weights: Optional dict mapping member names to weight tensors
        
        Returns:
            Scalar loss tensor
        """
        if effective_weights is None:
            effective_weights = {}
            for member in self.members:
                if not isinstance(member, DrumMember):
                    effective_weights[member.name] = member.weights
        
        cross_loss = torch.tensor(0.0)
        
        # Get all non-drum members
        pitched_members = [
            m for m in self.members
            if not isinstance(m, DrumMember) and m.name in effective_weights
        ]
        
        if len(pitched_members) < 2:
            return cross_loss
        
        # Find the maximum number of beats among all members
        max_beats = max(m.num_beats for m in pitched_members)
        
        # For each pair of members
        for i, member1 in enumerate(pitched_members):
            for member2 in pitched_members[i+1:]:
                weights1 = effective_weights[member1.name]
                weights2 = effective_weights[member2.name]
                
                # Get cross-dissonance matrix
                key = (member1.name, member2.name)
                D_cross = self.cross_dissonance_matrices.get(key)
                if D_cross is None:
                    # Try reverse order
                    key = (member2.name, member1.name)
                    D_cross = self.cross_dissonance_matrices.get(key)
                    if D_cross is None:
                        continue
                    # Transpose if we got the reverse
                    D_cross = D_cross.T
                
                # Apply activations
                activated1 = self._get_activated_weights(member1, weights1)
                activated2 = self._get_activated_weights(member2, weights2)
                
                # Calculate dissonance for each time slice
                # Align by beat index (assuming same total duration)
                # Normalize to common time grid
                for beat_idx in range(max_beats):
                    # Map beat to normalized time
                    time1 = beat_idx / member1.num_beats if member1.num_beats > 0 else 0
                    time2 = beat_idx / member2.num_beats if member2.num_beats > 0 else 0
                    
                    # Find the closest beat in each member
                    beat1 = min(int(time1 * member1.num_beats), member1.num_beats - 1)
                    beat2 = min(int(time2 * member2.num_beats), member2.num_beats - 1)
                    
                    w1 = activated1[:, beat1]
                    w2 = activated2[:, beat2]
                    
                    # Cross-dissonance: w1^T D_cross w2
                    cross_loss = cross_loss + torch.sum(w1 * (D_cross @ w2))
        
        return cross_loss
    
    def _get_activated_weights(
        self,
        member: BandMember,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Get activated weights based on member type.
        
        For PolyphonicMember: apply soft thresholding with sigmoid
        For MonophonicMember: use straight-through estimator
        
        Args:
            member: Band member
            weights: Raw weight tensor
        
        Returns:
            Activated weight tensor
        """
        if isinstance(member, PolyphonicMember):
            # Soft thresholding with sigmoid
            return torch.sigmoid(
                (weights - member.threshold) / member.soft_temperature
            )
        elif isinstance(member, MonophonicMember):
            # For monophonic, we need to process each beat separately
            activated = torch.zeros_like(weights)
            for beat in range(weights.shape[1]):
                activated[:, beat] = member.get_gumbel_sample(beat, training=True)
            return activated
        else:
            return weights
    
    def _calculate_range_penalty(
        self,
        member: BandMember,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Calculate range penalty for notes outside instrument range.
        
        Args:
            member: Band member
            weights: Weight tensor
        
        Returns:
            Scalar penalty tensor
        """
        penalty = torch.tensor(0.0)
        
        # Check each key
        for key_idx in range(member.num_keys):
            midi_key = member.get_midi_key(key_idx)
            if not member.instrument.is_key_in_range(midi_key):
                # Penalize weights for out-of-range keys
                penalty = penalty + weights[key_idx, :].sum()
        
        return penalty
    
    def _calculate_interval_jump_penalty(
        self,
        member: MonophonicMember,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Calculate interval jump penalty for large melodic leaps.
        
        Args:
            member: Monophonic member
            weights: Weight tensor
        
        Returns:
            Scalar penalty tensor
        """
        num_beats = weights.shape[1]
        if num_beats <= 1:
            return torch.tensor(0.0)
        
        penalty = torch.tensor(0.0)
        
        # For each pair of adjacent beats
        for beat in range(num_beats - 1):
            # Get gumbel samples (differentiable)
            w1 = member.get_gumbel_sample(beat, training=True)
            w2 = member.get_gumbel_sample(beat + 1, training=True)
            
            # Calculate expected key indices (soft)
            keys = torch.arange(member.num_keys, dtype=torch.float32)
            key1 = torch.sum(keys * w1)
            key2 = torch.sum(keys * w2)
            
            # Penalize large jumps (squared difference)
            jump_size = torch.abs(key2 - key1)
            penalty = penalty + jump_size ** 2
        
        return penalty / (num_beats - 1)  # Normalize by number of intervals


# ==================== VERIFICATION TESTS ====================

if __name__ == "__main__":
    print("=" * 60)
    print("LOSS FUNCTION VERIFICATION")
    print("=" * 60)
    
    from .band import PolyphonicMember, MonophonicMember, DrumMember
    from .tuning import TwelveTET
    
    # Test 1: Single member loss
    print("\n1. Single Member Loss Calculation:")
    piano = PolyphonicMember.piano(num_beats=4)
    loss_fn = LossFunction([piano])
    loss_fn.precompute_dissonance_matrices()
    
    total_loss, breakdown = loss_fn.calculate()
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Breakdown: {breakdown}")
    
    # Test 2: Multiple members
    print("\n2. Multiple Members Loss Calculation:")
    guitar = MonophonicMember.guitar(num_beats=4)
    bass = MonophonicMember.bass(num_beats=4)
    drums = DrumMember.standard_rock(num_beats=4)
    
    members = [piano, guitar, bass, drums]
    loss_fn = LossFunction(members)
    loss_fn.precompute_dissonance_matrices()
    
    total_loss, breakdown = loss_fn.calculate()
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Breakdown: {breakdown}")
    
    # Test 3: Within-member dissonance
    print("\n3. Within-Member Dissonance:")
    piano_loss = loss_fn.calculate_within_member(piano)
    print(f"   Piano within-member loss: {piano_loss.item():.4f}")
    
    # Test 4: Temporal dissonance
    print("\n4. Temporal Dissonance:")
    temporal_loss = loss_fn.calculate_temporal(piano)
    print(f"   Piano temporal loss: {temporal_loss.item():.4f}")
    
    # Test 5: Cross-member dissonance
    print("\n5. Cross-Member Dissonance:")
    cross_loss = loss_fn.calculate_cross_member()
    print(f"   Cross-member loss: {cross_loss.item():.4f}")
    
    # Test 6: Custom loss weights
    print("\n6. Custom Loss Weights:")
    custom_weights = {'within': 2.0, 'cross': 0.5, 'density': 5.0}
    loss_fn_weighted = LossFunction(members, loss_weights=custom_weights)
    loss_fn_weighted.precompute_dissonance_matrices()
    total_loss, breakdown = loss_fn_weighted.calculate()
    print(f"   Weighted total loss: {total_loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)
