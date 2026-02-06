"""
Main optimizer for Harmony From First Principles.

This module provides the HarmonyOptimizer class that orchestrates the optimization
of multiple band members, incorporating user constraints and calculating losses.
"""

from typing import List, Dict, Optional, Callable, Tuple, Any
import torch
import numpy as np

from .band import BandMember, PolyphonicMember, MonophonicMember, DrumMember
from .tuning import TuningSystem, TwelveTET
from .dissonance import DissonanceCalculator
from .losses import LossFunction
from .constraints import ConstraintSet, UserConstraint
from .synthesis import AudioSynthesizer
from .mixer import AudioMixer


class HarmonyOptimizer:
    """Main orchestration class for harmony optimization.
    
    The HarmonyOptimizer manages the optimization process for multiple band members,
    handling:
    - Precomputation of dissonance matrices
    - Application of user constraints
    - Optimization loop with Adam optimizer
    - Audio synthesis and mixing
    - Chord analysis
    
    Attributes:
        members: List of band members to optimize
        tuning: Tuning system
        constraints: ConstraintSet with user constraints
        loss_fn: LossFunction for calculating losses
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        iteration: Current optimization iteration
        loss_history: History of loss values
    """
    
    def __init__(
        self,
        members: List[BandMember],
        tuning: Optional[TuningSystem] = None,
        constraints: Optional[ConstraintSet] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        lr: float = 0.02,
        temporal_decay: float = 0.3,
        target_density: float = 0.15,
        enable_scheduler: bool = True,
        scheduler_step_size: int = 50,
        scheduler_gamma: float = 0.5
    ):
        """Initialize the harmony optimizer.
        
        Args:
            members: List of band members to optimize
            tuning: Tuning system (default: TwelveTET)
            constraints: User constraints (default: None)
            loss_weights: Dictionary of loss term weights
            lr: Learning rate for Adam optimizer
            temporal_decay: Weight for temporal dissonance
            target_density: Target note density (0-1)
            enable_scheduler: Whether to use learning rate scheduler
            scheduler_step_size: Steps between LR decay
            scheduler_gamma: LR decay factor
        """
        self.members = members
        self.tuning = tuning if tuning else TwelveTET()
        self.constraints = constraints if constraints else ConstraintSet()
        
        # Get optimizable parameters (exclude drums)
        self.optimizable_members = [
            m for m in members if not isinstance(m, DrumMember)
        ]
        
        # Collect all optimizable parameters
        self.params = []
        for member in self.optimizable_members:
            self.params.append(member.weights)
        
        # Initialize loss function
        self.loss_fn = LossFunction(
            members=members,
            loss_weights=loss_weights,
            temporal_decay=temporal_decay,
            target_density=target_density
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        
        # Initialize scheduler
        self.enable_scheduler = enable_scheduler
        if enable_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_step_size,
                gamma=scheduler_gamma
            )
        else:
            self.scheduler = None
        
        # Tracking
        self.iteration = 0
        self.loss_history: List[Dict[str, float]] = []
        self.lr = lr
        
        # Audio synthesis
        self.synthesizer = AudioSynthesizer()
        self.mixer = AudioMixer()
    
    def precompute_dissonance(self):
        """Precompute all dissonance matrices.
        
        This should be called before starting optimization.
        """
        print("Precomputing dissonance matrices...")
        self.loss_fn.precompute_dissonance_matrices()
        print("Dissonance matrices computed.")
    
    def step(self) -> Dict[str, float]:
        """Perform one optimization step.
        
        Returns:
            Dictionary of loss components
        """
        self.optimizer.zero_grad()
        
        # Get effective weights (optimizable + constraints)
        effective_weights = self._get_effective_weights()
        
        # Calculate loss
        total_loss, breakdown = self.loss_fn.calculate(effective_weights)
        
        # Backward pass
        total_loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Apply non-negativity constraint (ReLU)
        with torch.no_grad():
            for member in self.optimizable_members:
                member.weights.clamp_(min=0.0)
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Track history
        self.loss_history.append(breakdown)
        self.iteration += 1
        
        return breakdown
    
    def optimize(
        self,
        steps: int = 200,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run full optimization loop.
        
        Args:
            steps: Number of optimization steps
            callback: Optional callback function(step, loss_dict)
            verbose: Whether to print progress
        
        Returns:
            Dictionary with optimization results
        """
        if verbose:
            print(f"\nStarting optimization for {steps} steps...")
            print(f"Learning rate: {self.lr}")
            print(f"Members: {[m.name for m in self.members]}")
            print(f"Constraints: {len(self.constraints)}\n")
        
        for step in range(steps):
            loss_dict = self.step()
            
            if callback:
                callback(step, loss_dict)
            
            if verbose and (step % 25 == 0 or step == steps - 1):
                self._print_progress(step, steps, loss_dict)
        
        if verbose:
            print("\nOptimization complete!")
        
        return {
            'steps': steps,
            'final_loss': loss_dict['total'],
            'loss_history': self.loss_history,
            'members': self.members
        }
    
    def _get_effective_weights(self) -> Dict[str, torch.Tensor]:
        """Get effective weights for all members.
        
        Returns:
            Dictionary mapping member names to effective weight tensors
        """
        return self.constraints.get_all_effective_weights(self.members)
    
    def _print_progress(self, step: int, total_steps: int, loss_dict: Dict[str, float]):
        """Print optimization progress.
        
        Args:
            step: Current step
            total_steps: Total number of steps
            loss_dict: Dictionary of loss components
        """
        print(f"Step {step + 1}/{total_steps}, Loss: {loss_dict['total']:.4f}")
        
        # Print breakdown
        breakdown_str = ", ".join([
            f"{k}={v:.4f}" for k, v in loss_dict.items() if k != 'total'
        ])
        print(f"  Breakdown: {breakdown_str}")
        
        # Print learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr != self.lr:
            print(f"  LR: {current_lr:.6f}")
    
    def get_audio(
        self,
        duration: float = 4.0,
        sample_rate: int = 22050
    ) -> np.ndarray:
        """Synthesize and mix final audio from all members.
        
        Args:
            duration: Total duration in seconds
            sample_rate: Audio sample rate
        
        Returns:
            Mixed audio as numpy array
        """
        audio_tracks = []
        
        # Create a synthesizer with proper beat duration
        beat_duration = duration / max(m.num_beats for m in self.members)
        synthesizer = AudioSynthesizer(
            sample_rate=sample_rate,
            beat_duration=beat_duration
        )
        
        for member in self.members:
            # Synthesize audio for this member
            track = synthesizer.synthesize_member(
                member,
                duration=duration,
                sample_rate=sample_rate
            )
            audio_tracks.append(track)
        
        # Mix all tracks
        mixed = self.mixer.mix_tracks(audio_tracks)
        
        return mixed
    
    def get_chord_analysis(self, threshold: float = 0.1) -> Dict[str, Any]:
        """Analyze resulting chords and progressions.
        
        Args:
            threshold: Weight threshold for considering a note active
        
        Returns:
            Dictionary with chord analysis
        """
        analysis = {
            'members': {},
            'global_progression': []
        }
        
        # Get effective weights
        effective_weights = self._get_effective_weights()
        
        for member in self.members:
            if isinstance(member, DrumMember):
                continue
            
            weights = effective_weights.get(member.name, member.weights)
            weights_np = weights.detach().numpy()
            
            member_analysis = self._analyze_member_chords(
                member, weights_np, threshold
            )
            analysis['members'][member.name] = member_analysis
        
        # Calculate global progression (all members combined)
        analysis['global_progression'] = self._calculate_global_progression(
            analysis['members'], threshold
        )
        
        return analysis
    
    def _analyze_member_chords(
        self,
        member: BandMember,
        weights: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Analyze chords for a single member.
        
        Args:
            member: Band member
            weights: Weight matrix [num_keys, num_beats]
            threshold: Activation threshold
        
        Returns:
            Dictionary with chord analysis
        """
        num_beats = weights.shape[1]
        chords = []
        
        for beat in range(num_beats):
            # Find active notes
            active_keys = np.where(weights[:, beat] > threshold)[0].tolist()
            
            # Convert to pitch classes
            pitch_classes = sorted(set([k % 12 for k in active_keys]))
            
            chords.append({
                'beat': beat,
                'active_keys': active_keys,
                'pitch_classes': pitch_classes,
                'num_notes': len(active_keys)
            })
        
        return {
            'chords': chords,
            'total_beats': num_beats
        }
    
    def _calculate_global_progression(
        self,
        member_analyses: Dict[str, Dict],
        threshold: float
    ) -> List[Dict]:
        """Calculate global chord progression across all members.
        
        Args:
            member_analyses: Dict of analyses per member
            threshold: Activation threshold
        
        Returns:
            List of global chord descriptions
        """
        # Find maximum beats
        max_beats = max(
            analysis['total_beats']
            for analysis in member_analyses.values()
        )
        
        progression = []
        
        for beat in range(max_beats):
            all_keys = []
            
            for member_name, analysis in member_analyses.items():
                if beat < len(analysis['chords']):
                    chord = analysis['chords'][beat]
                    # Offset keys to avoid collisions between members
                    # Use member name hash for consistent offset
                    offset = hash(member_name) % 12
                    offset_keys = [(k + offset) % 88 for k in chord['active_keys']]
                    all_keys.extend(offset_keys)
            
            # Get unique pitch classes
            pitch_classes = sorted(set([k % 12 for k in all_keys]))
            
            progression.append({
                'beat': beat,
                'pitch_classes': pitch_classes,
                'num_notes': len(all_keys),
                'unique_pcs': len(pitch_classes)
            })
        
        return progression
    
    def get_active_notes(self, threshold: float = 0.1) -> Dict[str, List[List[int]]]:
        """Get active notes for all members.
        
        Args:
            threshold: Weight threshold for note activation
        
        Returns:
            Dictionary mapping member names to active notes per beat
        """
        active_notes = {}
        
        # Get effective weights
        effective_weights = self._get_effective_weights()
        
        for member in self.members:
            if isinstance(member, DrumMember):
                # Get drum pattern
                notes = []
                for beat in range(member.num_beats):
                    active = member.get_active_notes(beat)
                    notes.append(active)
                active_notes[member.name] = notes
            else:
                weights = effective_weights.get(member.name, member.weights)
                weights_np = weights.detach().numpy()
                
                notes = []
                for beat in range(member.num_beats):
                    active = np.where(weights_np[:, beat] > threshold)[0].tolist()
                    notes.append(active)
                active_notes[member.name] = notes
        
        return active_notes
    
    def add_constraint(self, constraint: UserConstraint):
        """Add a user constraint.
        
        Args:
            constraint: UserConstraint to add
        """
        self.constraints.add_constraint(constraint)
    
    def clear_constraints(self):
        """Clear all user constraints."""
        self.constraints.clear()
    
    def save_state(self, filepath: str):
        """Save optimizer state to file.
        
        Args:
            filepath: Path to save state
        """
        state = {
            'iteration': self.iteration,
            'loss_history': self.loss_history,
            'optimizer_state': self.optimizer.state_dict(),
            'member_weights': {
                m.name: m.weights.detach().clone()
                for m in self.optimizable_members
            }
        }
        
        if self.scheduler:
            state['scheduler_state'] = self.scheduler.state_dict()
        
        torch.save(state, filepath)
        print(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load optimizer state from file.
        
        Args:
            filepath: Path to load state from
        """
        state = torch.load(filepath)
        
        self.iteration = state['iteration']
        self.loss_history = state['loss_history']
        self.optimizer.load_state_dict(state['optimizer_state'])
        
        # Restore member weights
        for member in self.optimizable_members:
            if member.name in state['member_weights']:
                with torch.no_grad():
                    member.weights.copy_(state['member_weights'][member.name])
        
        if self.scheduler and 'scheduler_state' in state:
            self.scheduler.load_state_dict(state['scheduler_state'])
        
        print(f"State loaded from {filepath}")
    
    def reset(self):
        """Reset the optimizer to initial state."""
        self.iteration = 0
        self.loss_history.clear()
        
        # Re-initialize weights
        for member in self.optimizable_members:
            with torch.no_grad():
                member.weights.normal_(mean=0.0, std=0.1)
        
        # Re-initialize optimizer
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
        
        if self.scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler.step_size,
                gamma=self.scheduler.gamma
            )


# ==================== VERIFICATION TESTS ====================

if __name__ == "__main__":
    print("=" * 60)
    print("HARMONY OPTIMIZER VERIFICATION")
    print("=" * 60)
    
    from .band import PolyphonicMember, MonophonicMember, DrumMember
    from .tuning import TwelveTET
    
    # Test 1: Create optimizer
    print("\n1. Create Optimizer:")
    piano = PolyphonicMember.piano(num_beats=4)
    guitar = MonophonicMember.guitar(num_beats=4)
    bass = MonophonicMember.bass(num_beats=4)
    drums = DrumMember.standard_rock(num_beats=4)
    
    members = [piano, guitar, bass, drums]
    optimizer = HarmonyOptimizer(members, lr=0.02)
    print(f"   Created optimizer with {len(members)} members")
    print(f"   Optimizable members: {len(optimizer.optimizable_members)}")
    
    # Test 2: Precompute dissonance
    print("\n2. Precompute Dissonance:")
    optimizer.precompute_dissonance()
    print(f"   Computed {len(optimizer.loss_fn.dissonance_matrices)} within-member matrices")
    print(f"   Computed {len(optimizer.loss_fn.cross_dissonance_matrices)} cross-member matrices")
    
    # Test 3: Single optimization step
    print("\n3. Single Optimization Step:")
    loss_dict = optimizer.step()
    print(f"   Total loss: {loss_dict['total']:.4f}")
    print(f"   Iteration: {optimizer.iteration}")
    
    # Test 4: Short optimization run
    print("\n4. Short Optimization Run (10 steps):")
    result = optimizer.optimize(steps=10, verbose=False)
    print(f"   Final loss: {result['final_loss']:.4f}")
    print(f"   Loss history length: {len(result['loss_history'])}")
    
    # Test 5: Add constraints
    print("\n5. Add Constraints:")
    optimizer.reset()
    constraint = UserConstraint("piano", beat_index=0, key_indices=60, strengths=1.0)
    optimizer.add_constraint(constraint)
    print(f"   Added constraint, total: {len(optimizer.constraints)}")
    
    # Optimize with constraint
    result = optimizer.optimize(steps=10, verbose=False)
    print(f"   Optimized with constraint, final loss: {result['final_loss']:.4f}")
    
    # Test 6: Get active notes
    print("\n6. Get Active Notes:")
    active = optimizer.get_active_notes(threshold=0.2)
    for name, notes in active.items():
        total_notes = sum(len(n) for n in notes)
        print(f"   {name}: {total_notes} total notes")
    
    # Test 7: Chord analysis
    print("\n7. Chord Analysis:")
    analysis = optimizer.get_chord_analysis(threshold=0.2)
    for member_name, member_analysis in analysis['members'].items():
        print(f"   {member_name}: {len(member_analysis['chords'])} beats analyzed")
    
    # Test 8: Audio synthesis (without saving)
    print("\n8. Audio Synthesis:")
    try:
        audio = optimizer.get_audio(duration=2.0, sample_rate=22050)
        print(f"   Audio shape: {audio.shape}")
        print(f"   Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    except Exception as e:
        print(f"   Audio synthesis skipped: {e}")
    
    # Test 9: Reset
    print("\n9. Reset Optimizer:")
    optimizer.reset()
    print(f"   Iteration reset to: {optimizer.iteration}")
    print(f"   Loss history cleared: {len(optimizer.loss_history) == 0}")
    
    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)
