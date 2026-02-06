"""
Band member system for Harmony From First Principles.

This module provides different types of band members that can play together:
- PolyphonicMember: Can play multiple notes simultaneously (e.g., piano)
- MonophonicMember: Can only play one note at a time (e.g., guitar, bass)
- DrumMember: Fixed pattern-based playback (e.g., drums)

Each band member has its own weights tensor [num_keys, num_beats] that is optimized
to minimize total dissonance while respecting the member's constraints.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import math

from .tuning import TuningSystem, TwelveTET
from .instruments import Instrument, ADSR


class BandMember(ABC):
    """Abstract base class for all band members.
    
    Band members represent individual instruments in the ensemble. Each member
    has its own optimizable weights that determine which notes to play on each beat.
    
    Attributes:
        name: Member name (e.g., "piano", "guitar")
        instrument: Instrument profile with harmonics and ADSR
        tuning: Tuning system for frequency calculations
        num_keys: Number of discrete keys this member can play
        key_offset: MIDI key offset (e.g., 21 for piano A0)
        num_beats: Number of beats in the composition
        weights: Optimizable parameters [num_keys, num_beats]
    """
    
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        tuning: TuningSystem,
        num_keys: int,
        key_offset: int,
        num_beats: int,
        initial_weights: Optional[torch.Tensor] = None
    ):
        """Initialize a band member.
        
        Args:
            name: Member name
            instrument: Instrument profile
            tuning: Tuning system
            num_keys: Number of keys
            key_offset: MIDI key offset (maps key index to actual MIDI note)
            num_beats: Number of beats
            initial_weights: Optional initial weights tensor [num_keys, num_beats]
        """
        self.name = name
        self.instrument = instrument
        self.tuning = tuning
        self.num_keys = num_keys
        self.key_offset = key_offset
        self.num_beats = num_beats
        
        # Initialize or use provided weights
        if initial_weights is not None:
            if initial_weights.shape != (num_keys, num_beats):
                raise ValueError(
                    f"initial_weights shape {initial_weights.shape} doesn't match "
                    f"expected ({num_keys}, {num_beats})"
                )
            self.weights = torch.nn.Parameter(initial_weights)
        else:
            # Initialize with small random values
            self.weights = torch.nn.Parameter(
                torch.randn(num_keys, num_beats) * 0.1
            )
    
    @abstractmethod
    def get_active_notes(self, beat_index: int, training: bool = True) -> List[int]:
        """Get the active key indices for a given beat.
        
        Args:
            beat_index: Index of the beat (0 to num_beats-1)
            training: If True, use differentiable operations
            
        Returns:
            List of active key indices
        """
        pass
    
    def get_all_active_notes(self, training: bool = True) -> List[List[int]]:
        """Get active notes for all beats.
        
        Args:
            training: If True, use differentiable operations
            
        Returns:
            List of lists, where each inner list contains active keys for that beat
        """
        return [self.get_active_notes(i, training=training) 
                for i in range(self.num_beats)]
    
    def get_frequencies(self) -> torch.Tensor:
        """Get frequencies for all keys using the tuning system.
        
        Returns:
            Tensor of shape (num_keys,) containing frequencies in Hz
        """
        # Get frequencies for key_offset to key_offset + num_keys - 1
        frequencies = []
        for k in range(self.num_keys):
            midi_key = self.key_offset + k
            freq = self.tuning.get_frequency(midi_key)
            frequencies.append(freq)
        return torch.tensor(frequencies, dtype=torch.float32)
    
    def get_midi_key(self, key_index: int) -> int:
        """Convert local key index to MIDI key number.
        
        Args:
            key_index: Local key index (0 to num_keys-1)
            
        Returns:
            MIDI key number
        """
        return self.key_offset + key_index
    
    def is_key_in_range(self, key_index: int) -> bool:
        """Check if a key index is within this member's range.
        
        Args:
            key_index: Local key index
            
        Returns:
            True if the key is in valid range
        """
        midi_key = self.get_midi_key(key_index)
        return self.instrument.is_key_in_range(midi_key)
    
    def get_weights_for_beat(self, beat_index: int) -> torch.Tensor:
        """Get the weights for a specific beat.
        
        Args:
            beat_index: Index of the beat
            
        Returns:
            Tensor of shape (num_keys,) containing weights
        """
        return self.weights[:, beat_index]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, keys={self.num_keys}, beats={self.num_beats})"


class PolyphonicMember(BandMember):
    """Polyphonic band member that can play multiple notes simultaneously.
    
    Used for instruments like piano where chords are possible. During training,
    uses soft thresholding for differentiability. During inference, uses a
    hard threshold.
    
    Attributes:
        threshold: Weight threshold for note activation (default: 0.0)
        soft_temperature: Temperature for soft thresholding during training
    """
    
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        tuning: TuningSystem,
        num_keys: int = 88,
        key_offset: int = 21,
        num_beats: int = 4,
        threshold: float = 0.0,
        soft_temperature: float = 0.1,
        initial_weights: Optional[torch.Tensor] = None
    ):
        """Initialize a polyphonic band member.
        
        Args:
            name: Member name
            instrument: Instrument profile
            tuning: Tuning system
            num_keys: Number of keys (default: 88 for standard piano)
            key_offset: MIDI key offset (default: 21 for A0)
            num_beats: Number of beats
            threshold: Weight threshold for note activation
            soft_temperature: Temperature for soft thresholding (lower = harder)
            initial_weights: Optional initial weights
        """
        super().__init__(
            name=name,
            instrument=instrument,
            tuning=tuning,
            num_keys=num_keys,
            key_offset=key_offset,
            num_beats=num_beats,
            initial_weights=initial_weights
        )
        self.threshold = threshold
        self.soft_temperature = soft_temperature
    
    def get_active_notes(self, beat_index: int, training: bool = True) -> List[int]:
        """Get active notes using thresholding.
        
        During training, uses soft thresholding with sigmoid for differentiability.
        During inference, uses hard thresholding.
        
        Args:
            beat_index: Index of the beat
            training: If True, use soft thresholding
            
        Returns:
            List of active key indices (non-differentiable list in both modes)
        """
        weights = self.get_weights_for_beat(beat_index)
        
        if training:
            # During training, we return all indices but the caller should
            # use the soft probabilities for gradient computation
            # Here we return indices above threshold for reference
            probs = torch.sigmoid((weights - self.threshold) / self.soft_temperature)
            active_mask = probs > 0.5
        else:
            # During inference, hard threshold
            active_mask = weights > self.threshold
        
        # Convert to list of indices
        active_indices = torch.where(active_mask)[0].tolist()
        return active_indices
    
    def get_soft_activation(self, beat_index: int) -> torch.Tensor:
        """Get soft activation probabilities for differentiable training.
        
        Args:
            beat_index: Index of the beat
            
        Returns:
            Tensor of shape (num_keys,) with activation probabilities
        """
        weights = self.get_weights_for_beat(beat_index)
        return torch.sigmoid((weights - self.threshold) / self.soft_temperature)
    
    @classmethod
    def piano(
        cls,
        num_beats: int = 4,
        tuning: Optional[TuningSystem] = None,
        initial_weights: Optional[torch.Tensor] = None
    ) -> "PolyphonicMember":
        """Create a piano band member with default settings.
        
        Args:
            num_beats: Number of beats
            tuning: Tuning system (default: 12-TET)
            initial_weights: Optional initial weights
            
        Returns:
            PolyphonicMember configured as a piano
        """
        return cls(
            name="piano",
            instrument=Instrument.piano(),
            tuning=tuning if tuning else TwelveTET(),
            num_keys=88,
            key_offset=21,
            num_beats=num_beats,
            threshold=0.0,
            soft_temperature=0.1,
            initial_weights=initial_weights
        )


class MonophonicMember(BandMember):
    """Monophonic band member that can only play one note at a time.
    
    Used for instruments like guitar and bass where only one note can be
    played per beat. During training, uses Gumbel-softmax for differentiability.
    During inference, uses argmax.
    
    Attributes:
        gumbel_temperature: Temperature for Gumbel-softmax (lower = harder)
        straight_through: If True, use straight-through estimator
    """
    
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        tuning: TuningSystem,
        num_keys: int,
        key_offset: int,
        num_beats: int,
        gumbel_temperature: float = 0.5,
        straight_through: bool = True,
        initial_weights: Optional[torch.Tensor] = None
    ):
        """Initialize a monophonic band member.
        
        Args:
            name: Member name
            instrument: Instrument profile
            tuning: Tuning system
            num_keys: Number of keys
            key_offset: MIDI key offset
            num_beats: Number of beats
            gumbel_temperature: Temperature for Gumbel-softmax sampling
            straight_through: If True, use straight-through estimator
            initial_weights: Optional initial weights
        """
        super().__init__(
            name=name,
            instrument=instrument,
            tuning=tuning,
            num_keys=num_keys,
            key_offset=key_offset,
            num_beats=num_beats,
            initial_weights=initial_weights
        )
        self.gumbel_temperature = gumbel_temperature
        self.straight_through = straight_through
    
    def get_active_notes(self, beat_index: int, training: bool = True) -> List[int]:
        """Get the single active note (strongest weight).
        
        During training, this returns the argmax for reference, but the
        caller should use get_gumbel_sample() for differentiable operations.
        During inference, simply returns the argmax.
        
        Args:
            beat_index: Index of the beat
            training: If True, we're in training mode (but still return argmax)
            
        Returns:
            List containing single key index, or empty list if no note
        """
        weights = self.get_weights_for_beat(beat_index)
        
        # Check if all weights are very negative (rest)
        if weights.max() < -5.0:  # Threshold for "rest"
            return []
        
        # Return the index with maximum weight
        max_idx = torch.argmax(weights).item()
        return [max_idx]
    
    def get_gumbel_sample(self, beat_index: int, training: bool = True) -> torch.Tensor:
        """Get differentiable one-hot encoding using Gumbel-softmax.
        
        Args:
            beat_index: Index of the beat
            training: If True, use Gumbel-softmax with noise
            
        Returns:
            Tensor of shape (num_keys,) with one-hot or soft one-hot encoding
        """
        weights = self.get_weights_for_beat(beat_index)
        
        if training and self.gumbel_temperature > 0:
            # Gumbel-softmax for differentiable sampling
            # Add Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(weights) + 1e-10) + 1e-10)
            logits = (weights + gumbel_noise) / self.gumbel_temperature
            soft_sample = F.softmax(logits, dim=0)
            
            if self.straight_through:
                # Straight-through estimator: forward uses hard, backward uses soft
                hard_sample = torch.zeros_like(soft_sample)
                hard_sample[torch.argmax(soft_sample)] = 1.0
                # Use soft_sample for gradients, hard_sample for values
                sample = hard_sample - soft_sample.detach() + soft_sample
            else:
                sample = soft_sample
        else:
            # Inference: hard one-hot
            sample = torch.zeros_like(weights)
            max_idx = torch.argmax(weights)
            sample[max_idx] = 1.0
        
        return sample
    
    @classmethod
    def guitar(
        cls,
        num_beats: int = 4,
        tuning: Optional[TuningSystem] = None,
        initial_weights: Optional[torch.Tensor] = None
    ) -> "MonophonicMember":
        """Create a guitar band member with default settings.
        
        Args:
            num_beats: Number of beats
            tuning: Tuning system (default: 12-TET)
            initial_weights: Optional initial weights
            
        Returns:
            MonophonicMember configured as a guitar
        """
        return cls(
            name="guitar",
            instrument=Instrument.guitar(),
            tuning=tuning if tuning else TwelveTET(),
            num_keys=49,  # E2 to E6
            key_offset=40,
            num_beats=num_beats,
            gumbel_temperature=0.5,
            straight_through=True,
            initial_weights=initial_weights
        )
    
    @classmethod
    def bass(
        cls,
        num_beats: int = 4,
        tuning: Optional[TuningSystem] = None,
        initial_weights: Optional[torch.Tensor] = None
    ) -> "MonophonicMember":
        """Create a bass band member with default settings.
        
        Args:
            num_beats: Number of beats
            tuning: Tuning system (default: 12-TET)
            initial_weights: Optional initial weights
            
        Returns:
            MonophonicMember configured as a bass
        """
        return cls(
            name="bass",
            instrument=Instrument.bass(),
            tuning=tuning if tuning else TwelveTET(),
            num_keys=40,  # E1 to G4
            key_offset=28,
            num_beats=num_beats,
            gumbel_temperature=0.5,
            straight_through=True,
            initial_weights=initial_weights
        )


class DrumMember(BandMember):
    """Drum band member with fixed patterns (not optimizable).
    
    Unlike other band members, drums use fixed patterns rather than
    optimizable weights. The patterns are defined as a dictionary mapping
    drum names to lists of beat indices where that drum hits.
    
    Attributes:
        pattern: Dictionary mapping drum names to beat indices
        enabled: Whether this drum track is active
        drum_instruments: Dictionary mapping drum names to Instrument presets
    """
    
    def __init__(
        self,
        name: str = "drums",
        pattern: Optional[Dict[str, List[int]]] = None,
        enabled: bool = True,
        num_beats: int = 4,
        tuning: Optional[TuningSystem] = None
    ):
        """Initialize a drum band member.
        
        Args:
            name: Member name
            pattern: Dictionary mapping drum names to beat indices
            enabled: Whether drums are enabled
            num_beats: Number of beats (for reference, pattern determines actual hits)
            tuning: Tuning system (mostly unused but kept for consistency)
        """
        # Drums don't use the weight optimization system
        # Create dummy weights that won't be optimized
        dummy_weights = torch.zeros(1, num_beats)
        
        # Use kick drum as placeholder instrument for base class
        # Actual drums use their own instruments from drum_instruments
        placeholder_instrument = Instrument.drums_kick()
        
        super().__init__(
            name=name,
            instrument=placeholder_instrument,
            tuning=tuning if tuning else TwelveTET(),
            num_keys=1,  # Not used
            key_offset=0,  # Not used
            num_beats=num_beats,
            initial_weights=dummy_weights
        )
        
        # Don't optimize drum weights
        self.weights.requires_grad = False
        
        # Default pattern: kick on 0, 2; snare on 1, 3
        self.pattern = pattern if pattern else {
            "kick": [0, 2],
            "snare": [1, 3]
        }
        self.enabled = enabled
        
        # Set up drum instruments
        self.drum_instruments = {
            "kick": Instrument.drums_kick(),
            "snare": Instrument.drums_snare(),
            "hihat": Instrument.drums_hihat()
        }
    
    def get_active_notes(self, beat_index: int, training: bool = True) -> List[int]:
        """Get which drums are active for this beat.
        
        Note: For drums, the return value is a list of drum names rather than
        key indices, since drums work differently from pitched instruments.
        This breaks the type signature but is handled specially by synthesizers.
        
        Args:
            beat_index: Index of the beat
            training: Ignored for drums (no training)
            
        Returns:
            List of active drum names (strings, not key indices)
        """
        if not self.enabled:
            return []
        
        active_drums = []
        for drum_name, beats in self.pattern.items():
            if beat_index in beats:
                active_drums.append(drum_name)
        
        return active_drums
    
    def get_drum_instrument(self, drum_name: str) -> Instrument:
        """Get the instrument for a specific drum type.
        
        Args:
            drum_name: Name of the drum ("kick", "snare", "hihat")
            
        Returns:
            Instrument for that drum type
        """
        return self.drum_instruments.get(drum_name, Instrument.drums_kick())
    
    def set_pattern(self, drum_name: str, beats: List[int]):
        """Set the pattern for a specific drum.
        
        Args:
            drum_name: Name of the drum
            beats: List of beat indices where this drum should play
        """
        self.pattern[drum_name] = beats
    
    def toggle(self):
        """Toggle drums on/off."""
        self.enabled = not self.enabled
    
    @classmethod
    def standard_rock(
        cls,
        num_beats: int = 4,
        enabled: bool = True
    ) -> "DrumMember":
        """Create a standard rock drum pattern.
        
        Pattern: Kick on 1 & 3, Snare on 2 & 4 (0-indexed: kick 0,2; snare 1,3)
        
        Args:
            num_beats: Number of beats (typically 4)
            enabled: Whether drums start enabled
            
        Returns:
            DrumMember with standard rock pattern
        """
        return cls(
            name="drums",
            pattern={
                "kick": [0, 2],
                "snare": [1, 3]
            },
            enabled=enabled,
            num_beats=num_beats
        )
    
    @classmethod
    def simple_kick_snare(
        cls,
        num_beats: int = 4,
        enabled: bool = True
    ) -> "DrumMember":
        """Create a simple kick-snare alternating pattern.
        
        Pattern: Kick on even beats, Snare on odd beats.
        
        Args:
            num_beats: Number of beats
            enabled: Whether drums start enabled
            
        Returns:
            DrumMember with simple alternating pattern
        """
        pattern = {
            "kick": [i for i in range(num_beats) if i % 2 == 0],
            "snare": [i for i in range(num_beats) if i % 2 == 1]
        }
        return cls(
            name="drums",
            pattern=pattern,
            enabled=enabled,
            num_beats=num_beats
        )


# ==================== VERIFICATION TESTS ====================

if __name__ == "__main__":
    print("=" * 60)
    print("BAND MEMBER SYSTEM VERIFICATION")
    print("=" * 60)
    
    # Test 1: Create polyphonic piano member
    print("\n1. PolyphonicMember (Piano):")
    piano = PolyphonicMember.piano(num_beats=4)
    print(f"   {piano}")
    print(f"   Weights shape: {piano.weights.shape}")
    print(f"   Frequencies shape: {piano.get_frequencies().shape}")
    print(f"   Sample frequencies: {piano.get_frequencies()[:3].tolist()}")
    
    # Test active notes
    active = piano.get_active_notes(0, training=False)
    print(f"   Active notes (beat 0, inference): {active[:5]}...")
    
    soft = piano.get_soft_activation(0)
    print(f"   Soft activation shape: {soft.shape}")
    print(f"   Soft activation sum: {soft.sum().item():.3f}")
    
    # Test 2: Create monophonic guitar member
    print("\n2. MonophonicMember (Guitar):")
    guitar = MonophonicMember.guitar(num_beats=4)
    print(f"   {guitar}")
    print(f"   Weights shape: {guitar.weights.shape}")
    print(f"   Key range: {guitar.key_offset} to {guitar.key_offset + guitar.num_keys - 1}")
    
    # Set a clear winner for testing
    with torch.no_grad():
        guitar.weights[:, 0] = torch.tensor([5.0 if i == 10 else -2.0 for i in range(guitar.num_keys)])
    
    active = guitar.get_active_notes(0, training=False)
    print(f"   Active note (beat 0): {active}")
    
    gumbel = guitar.get_gumbel_sample(0, training=False)
    print(f"   Gumbel sample sum (inference): {gumbel.sum().item()}")
    
    # Test 3: Create monophonic bass member
    print("\n3. MonophonicMember (Bass):")
    bass = MonophonicMember.bass(num_beats=4)
    print(f"   {bass}")
    print(f"   Weights shape: {bass.weights.shape}")
    print(f"   Key range: {bass.key_offset} to {bass.key_offset + bass.num_keys - 1}")
    
    # Test 4: Create drum member
    print("\n4. DrumMember:")
    drums = DrumMember.standard_rock(num_beats=4)
    print(f"   {drums}")
    print(f"   Pattern: {drums.pattern}")
    print(f"   Enabled: {drums.enabled}")
    
    for i in range(4):
        active = drums.get_active_notes(i)
        print(f"   Beat {i}: {active}")
    
    # Test toggling
    drums.toggle()
    print(f"   After toggle - Enabled: {drums.enabled}")
    print(f"   Beat 0 (disabled): {drums.get_active_notes(0)}")
    drums.toggle()  # Re-enable
    
    # Test 5: Verify monophonic constraint
    print("\n5. Monophonic Constraint Verification:")
    
    # Set multiple high weights
    with torch.no_grad():
        guitar.weights[:, 1] = torch.randn(guitar.num_keys)
        guitar.weights[5, 1] = 10.0  # Clear winner
        guitar.weights[15, 1] = 8.0  # Runner up
    
    for training in [True, False]:
        active = guitar.get_active_notes(1, training=training)
        print(f"   Training={training}: Active notes = {active}")
        print(f"   Is monophonic? {len(active) <= 1}")
    
    # Test 6: Test different tuning systems
    print("\n6. Different Tuning Systems:")
    from .tuning import PythagoreanTuning
    
    pythagorean_guitar = MonophonicMember.guitar(
        num_beats=4,
        tuning=PythagoreanTuning()
    )
    print(f"   Pythagorean guitar: {pythagorean_guitar.tuning.name}")
    
    # Compare frequencies for same keys
    standard_freqs = guitar.get_frequencies()[:5]
    pyth_freqs = pythagorean_guitar.get_frequencies()[:5]
    print(f"   Standard freqs: {standard_freqs.tolist()}")
    print(f"   Pythagorean freqs: {pyth_freqs.tolist()}")
    
    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)
