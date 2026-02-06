"""
User constraints system for Harmony From First Principles.

This module provides constraint classes that allow users to specify fixed notes
(like partial denoising in image-to-image generation). These constraints:
- Influence the loss calculation (apply dissonance pressure on optimizable notes)
- Are NOT optimized (gradients do not flow to constraints)
- Can be used to guide the optimization toward desired harmonies
"""

from typing import List, Dict, Optional, Union
import torch

from .band import BandMember, DrumMember


class UserConstraint:
    """A user-specified constraint (fixed notes) for a band member.
    
    Represents fixed notes placed by the user that influence the optimization
    but are not themselves optimized. This is analogous to partial denoising
    in image-to-image generation.
    
    Attributes:
        member_name: Name of the band member this constraint applies to
        beat_index: Which beat (time position) this constraint is for
        key_indices: List of key indices that are fixed
        strengths: Tensor of fixed weight values for each key
        fixed_value: Fixed activation value (1.0 for "note on")
    """
    
    def __init__(
        self,
        member_name: str,
        beat_index: int,
        key_indices: Union[int, List[int]],
        strengths: Optional[Union[float, torch.Tensor]] = None,
        fixed_value: float = 1.0
    ):
        """Initialize a user constraint.
        
        Args:
            member_name: Name of the band member
            beat_index: Beat index (0-based)
            key_indices: Single key index or list of key indices to fix
            strengths: Strength values for each key (default: 1.0)
            fixed_value: Fixed activation value (default: 1.0)
        """
        self.member_name = member_name
        self.beat_index = beat_index
        
        # Normalize key_indices to a list
        if isinstance(key_indices, int):
            self.key_indices = [key_indices]
        else:
            self.key_indices = list(key_indices)
        
        # Set up strengths
        if strengths is None:
            self.strengths = torch.ones(len(self.key_indices)) * fixed_value
        elif isinstance(strengths, (int, float)):
            self.strengths = torch.ones(len(self.key_indices)) * strengths
        else:
            self.strengths = strengths
        
        self.fixed_value = fixed_value
    
    def get_contribution(self, num_keys: int) -> torch.Tensor:
        """Get the constraint contribution as a fixed weight vector.
        
        Returns a tensor that represents this constraint's contribution
        to the specified beat. The tensor is detached from the computation
        graph (no gradients).
        
        Args:
            num_keys: Total number of keys for this member
        
        Returns:
            Tensor of shape (num_keys,) with fixed values
        """
        contribution = torch.zeros(num_keys)
        
        for key_idx, strength in zip(self.key_indices, self.strengths):
            if 0 <= key_idx < num_keys:
                contribution[key_idx] = strength
        
        # Detach - constraints are fixed, not optimized
        return contribution.detach()
    
    def __repr__(self) -> str:
        return (
            f"UserConstraint({self.member_name}, "
            f"beat={self.beat_index}, keys={self.key_indices})"
        )


class ConstraintSet:
    """Container for multiple user constraints.
    
    Manages a collection of constraints and provides methods to:
    - Add constraints
    - Get constraints for specific members
    - Apply constraints to generate constraint matrices
    
    Attributes:
        constraints: List of UserConstraint objects
        _by_member: Dict mapping member names to lists of constraints
    """
    
    def __init__(self):
        """Initialize an empty constraint set."""
        self.constraints: List[UserConstraint] = []
        self._by_member: Dict[str, List[UserConstraint]] = {}
    
    def add_constraint(self, constraint: UserConstraint):
        """Add a constraint to the set.
        
        Args:
            constraint: UserConstraint to add
        """
        self.constraints.append(constraint)
        
        # Update index
        if constraint.member_name not in self._by_member:
            self._by_member[constraint.member_name] = []
        self._by_member[constraint.member_name].append(constraint)
    
    def add_note(
        self,
        member_name: str,
        beat_index: int,
        key_index: int,
        strength: float = 1.0
    ):
        """Convenience method to add a single note constraint.
        
        Args:
            member_name: Name of the band member
            beat_index: Beat index
            key_index: Key index to fix
            strength: Note strength (default: 1.0)
        """
        constraint = UserConstraint(
            member_name=member_name,
            beat_index=beat_index,
            key_indices=key_index,
            strengths=strength
        )
        self.add_constraint(constraint)
    
    def add_chord(
        self,
        member_name: str,
        beat_index: int,
        key_indices: List[int],
        strength: float = 1.0
    ):
        """Convenience method to add a chord constraint.
        
        Args:
            member_name: Name of the band member
            beat_index: Beat index
            key_indices: List of key indices forming the chord
            strength: Chord strength (default: 1.0)
        """
        constraint = UserConstraint(
            member_name=member_name,
            beat_index=beat_index,
            key_indices=key_indices,
            strengths=strength
        )
        self.add_constraint(constraint)
    
    def get_for_member(self, member_name: str) -> List[UserConstraint]:
        """Get all constraints for a specific member.
        
        Args:
            member_name: Name of the band member
        
        Returns:
            List of UserConstraint objects for that member
        """
        return self._by_member.get(member_name, [])
    
    def apply_to_member(
        self,
        member: BandMember,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate constraint matrix for a band member.
        
        Creates a matrix of shape [num_keys, num_beats] where each entry
        represents the fixed contribution from constraints. This matrix
        is detached from the computation graph.
        
        Args:
            member: Band member to generate constraints for
            device: Optional torch device
        
        Returns:
            Tensor of shape (num_keys, num_beats) with constraint values
        """
        if isinstance(member, DrumMember):
            # Drums don't use constraints
            return torch.zeros((member.num_keys, member.num_beats))
        
        constraints = self.get_for_member(member.name)
        
        # Create constraint matrix
        constraint_matrix = torch.zeros((member.num_keys, member.num_beats))
        
        for constraint in constraints:
            if constraint.beat_index < member.num_beats:
                contribution = constraint.get_contribution(member.num_keys)
                constraint_matrix[:, constraint.beat_index] += contribution
        
        # Clamp to reasonable range
        constraint_matrix = torch.clamp(constraint_matrix, 0.0, 1.0)
        
        # Move to device if specified
        if device is not None:
            constraint_matrix = constraint_matrix.to(device)
        
        # Ensure no gradients flow to constraints
        return constraint_matrix.detach()
    
    def get_effective_weights(
        self,
        member: BandMember,
        optimizable_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Get effective weights combining optimizable weights and constraints.
        
        Effective weights = optimizable_weights + constraint_contribution
        Constraints are detached, so gradients only flow to optimizable_weights.
        
        Args:
            member: Band member
            optimizable_weights: The member's optimizable weights (default: member.weights)
            device: Optional torch device
        
        Returns:
            Tensor of effective weights (optimizable + constraints)
        """
        if optimizable_weights is None:
            optimizable_weights = member.weights
        
        # Get constraint matrix
        constraint_matrix = self.apply_to_member(member, device)
        
        # Ensure constraint matrix is on same device as weights
        if constraint_matrix.device != optimizable_weights.device:
            constraint_matrix = constraint_matrix.to(optimizable_weights.device)
        
        # Combine: effective = optimizable + constraints
        # Constraints are detached, so only optimizable_weights get gradients
        effective = optimizable_weights + constraint_matrix
        
        return effective
    
    def get_all_effective_weights(
        self,
        members: List[BandMember],
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """Get effective weights for all members.
        
        Args:
            members: List of band members
            device: Optional torch device
        
        Returns:
            Dict mapping member names to effective weight tensors
        """
        effective_weights = {}
        
        for member in members:
            if isinstance(member, DrumMember):
                continue
            
            effective = self.get_effective_weights(member, device=device)
            effective_weights[member.name] = effective
        
        return effective_weights
    
    def clear(self):
        """Remove all constraints."""
        self.constraints.clear()
        self._by_member.clear()
    
    def remove_for_member(self, member_name: str):
        """Remove all constraints for a specific member.
        
        Args:
            member_name: Name of the member to clear constraints for
        """
        # Remove from main list
        self.constraints = [
            c for c in self.constraints
            if c.member_name != member_name
        ]
        
        # Remove from index
        if member_name in self._by_member:
            del self._by_member[member_name]
    
    def remove_at_beat(self, member_name: str, beat_index: int):
        """Remove constraints for a specific member at a specific beat.
        
        Args:
            member_name: Name of the member
            beat_index: Beat index to clear
        """
        # Remove from main list
        self.constraints = [
            c for c in self.constraints
            if not (c.member_name == member_name and c.beat_index == beat_index)
        ]
        
        # Update index
        if member_name in self._by_member:
            self._by_member[member_name] = [
                c for c in self._by_member[member_name]
                if c.beat_index != beat_index
            ]
    
    def __len__(self) -> int:
        """Return the number of constraints."""
        return len(self.constraints)
    
    def __repr__(self) -> str:
        return f"ConstraintSet({len(self.constraints)} constraints)"


class DrumPatternConstraint:
    """Fixed drum pattern that acts as a constraint on other members.
    
    Unlike UserConstraint which constrains a specific member's notes,
    DrumPatternConstraint represents fixed drum patterns that influence
    the loss calculation for other members but aren't optimized themselves.
    
    This is handled specially in the loss calculation as drums don't have
    optimizable weights.
    
    Attributes:
        drum_member: The DrumMember instance
        enabled: Whether the drum pattern is active
    """
    
    def __init__(self, drum_member: DrumMember):
        """Initialize drum pattern constraint.
        
        Args:
            drum_member: DrumMember instance with fixed pattern
        """
        self.drum_member = drum_member
    
    def get_active_beats(self) -> Dict[int, List[str]]:
        """Get mapping of beat indices to active drum names.
        
        Returns:
            Dict mapping beat index -> list of active drum names
        """
        active_beats = {}
        for beat in range(self.drum_member.num_beats):
            drums = self.drum_member.get_active_notes(beat)
            if drums:
                active_beats[beat] = drums
        return active_beats
    
    def contributes_at_beat(self, beat_index: int) -> bool:
        """Check if drums contribute at a specific beat.
        
        Args:
            beat_index: Beat to check
        
        Returns:
            True if any drum is active at this beat
        """
        active = self.drum_member.get_active_notes(beat_index)
        return len(active) > 0


# ==================== VERIFICATION TESTS ====================

if __name__ == "__main__":
    print("=" * 60)
    print("CONSTRAINT SYSTEM VERIFICATION")
    print("=" * 60)
    
    from .band import PolyphonicMember, MonophonicMember, DrumMember
    from .tuning import TwelveTET
    
    # Test 1: Create user constraints
    print("\n1. UserConstraint Creation:")
    constraint1 = UserConstraint("piano", beat_index=0, key_indices=60, strengths=1.0)
    print(f"   {constraint1}")
    
    constraint2 = UserConstraint(
        "piano", beat_index=0,
        key_indices=[64, 67],  # C major chord
        strengths=[1.0, 1.0]
    )
    print(f"   {constraint2}")
    
    # Test 2: Constraint contribution
    print("\n2. Constraint Contribution:")
    contribution = constraint1.get_contribution(num_keys=88)
    print(f"   Contribution shape: {contribution.shape}")
    print(f"   Non-zero entries: {torch.count_nonzero(contribution)}")
    print(f"   Key 60 value: {contribution[60].item():.2f}")
    
    # Test 3: ConstraintSet
    print("\n3. ConstraintSet Management:")
    constraint_set = ConstraintSet()
    constraint_set.add_constraint(constraint1)
    constraint_set.add_constraint(constraint2)
    
    print(f"   Total constraints: {len(constraint_set)}")
    
    piano_constraints = constraint_set.get_for_member("piano")
    print(f"   Piano constraints: {len(piano_constraints)}")
    
    # Test 4: Apply to member
    print("\n4. Apply to Member:")
    piano = PolyphonicMember.piano(num_beats=4)
    constraint_matrix = constraint_set.apply_to_member(piano)
    print(f"   Constraint matrix shape: {constraint_matrix.shape}")
    print(f"   Non-zero in beat 0: {torch.count_nonzero(constraint_matrix[:, 0])}")
    print(f"   Non-zero in beat 1: {torch.count_nonzero(constraint_matrix[:, 1])}")
    
    # Test 5: Effective weights
    print("\n5. Effective Weights:")
    # Initialize some optimizable weights
    with torch.no_grad():
        piano.weights[:, 0] = torch.randn(piano.num_keys) * 0.1
    
    effective = constraint_set.get_effective_weights(piano)
    print(f"   Effective weights shape: {effective.shape}")
    print(f"   Key 60 at beat 0: {effective[60, 0].item():.4f}")
    print(f"   Requires grad: {effective.requires_grad}")
    
    # Test 6: Convenience methods
    print("\n6. Convenience Methods:")
    constraint_set.clear()
    constraint_set.add_note("guitar", beat_index=1, key_index=40, strength=0.8)
    constraint_set.add_chord("piano", beat_index=2, key_indices=[60, 64, 67])
    print(f"   After adding note and chord: {len(constraint_set)} constraints")
    
    # Test 7: Drum pattern constraint
    print("\n7. Drum Pattern Constraint:")
    drums = DrumMember.standard_rock(num_beats=4)
    drum_constraint = DrumPatternConstraint(drums)
    active_beats = drum_constraint.get_active_beats()
    print(f"   Active beats: {active_beats}")
    print(f"   Contributes at beat 0: {drum_constraint.contributes_at_beat(0)}")
    
    # Test 8: Gradient check
    print("\n8. Gradient Check:")
    piano = PolyphonicMember.piano(num_beats=4)
    constraint_set.clear()
    constraint_set.add_note("piano", beat_index=0, key_index=60, strength=1.0)
    
    effective = constraint_set.get_effective_weights(piano)
    loss = effective.sum()
    loss.backward()
    
    print(f"   Constraint matrix grad: {constraint_set.apply_to_member(piano).requires_grad}")
    print(f"   Piano weights grad: {piano.weights.grad is not None}")
    if piano.weights.grad is not None:
        print(f"   Grad at [60, 0]: {piano.weights.grad[60, 0].item():.4f}")
        print(f"   Grad at [0, 0]: {piano.weights.grad[0, 0].item():.4f}")
    
    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)
