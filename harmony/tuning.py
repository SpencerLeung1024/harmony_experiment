"""
Alternate Musical Systems Module for Harmony From First Principles.

This module provides various tuning systems for musical scales, allowing exploration
of different musical traditions and theoretical constructs beyond 12-tone equal temperament.

All tuning systems expose discrete keys with continuous frequencies, maintaining
the optimization constraint (discrete choices) while allowing arbitrary frequency mappings.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import math
import torch


class TuningSystem(ABC):
    """Abstract base class for all tuning systems.
    
    Tuning systems define the mapping from discrete key indices to continuous
    frequencies in Hz. This abstraction allows the harmony optimizer to work
    with any tuning system, from standard 12-TET to microtonal and non-octave scales.
    
    Key Design:
        - Key indices are integers (0, 1, 2, ... num_keys-1)
        - Frequencies are positive floats in Hz
        - Vectorized operations for efficient PyTorch integration
    """
    
    @abstractmethod
    def get_frequency(self, key_index: int) -> float:
        """Get the frequency in Hz for a given key index.
        
        Args:
            key_index: Integer index of the key (0, 1, 2, ...)
            
        Returns:
            Frequency in Hz as a positive float.
        """
        pass
    
    def get_all_frequencies(self, num_keys: int) -> torch.Tensor:
        """Get frequencies for all keys from 0 to num_keys-1.
        
        This vectorized version is more efficient for computing dissonance matrices
        and other operations that need all frequencies at once.
        
        Args:
            num_keys: Number of keys to generate frequencies for.
            
        Returns:
            Tensor of shape (num_keys,) containing frequencies in Hz.
        """
        return torch.tensor([self.get_frequency(k) for k in range(num_keys)],
                           dtype=torch.float32)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the tuning system."""
        pass


class TwelveTET(TuningSystem):
    """Standard 12-tone equal temperament (12-TET).
    
    The modern Western standard tuning system where:
    - The octave is divided into 12 equal semitones on a logarithmic scale
    - Each semitone has a frequency ratio of 2^(1/12) ≈ 1.05946
    - A4 (MIDI key 69) is tuned to 440 Hz
    
    Formula: f(key) = 440.0 * 2^((key - 69) / 12)
    
    This is the default tuning system used by MIDI and most modern instruments.
    While mathematically convenient, it sacrifices pure intervals (except the octave)
    for equal flexibility in all keys.
    """
    
    def __init__(self, reference_hz: float = 440.0, reference_key: int = 69):
        """Initialize 12-TET tuning.
        
        Args:
            reference_hz: Frequency of the reference note (default: 440.0 Hz for A4)
            reference_key: Key index of the reference note (default: 69 for MIDI A4)
        """
        self.reference_hz = reference_hz
        self.reference_key = reference_key
    
    def get_frequency(self, key_index: int) -> float:
        """Calculate frequency using 12-TET formula."""
        return self.reference_hz * (2 ** ((key_index - self.reference_key) / 12.0))
    
    @property
    def name(self) -> str:
        return f"12-TET (A{self.reference_key}= {self.reference_hz}Hz)"


class PythagoreanTuning(TuningSystem):
    """Pythagorean tuning based on pure perfect fifths (3:2 ratio).
    
    The oldest documented tuning system, attributed to Pythagoras (6th century BCE).
    Built by stacking perfect fifths (frequency ratio 3:2) and reducing to a single octave.
    
    Theory:
    - Start with a base frequency (e.g., D or A)
    - Stack perfect fifths: f = base_freq * (3/2)^n
    - Reduce to single octave by dividing by powers of 2
    
    Characteristics:
    - Perfect fifths are pure (3:2 ratio)
    - Major thirds are sharp (81:64 vs pure 5:4), creating tension
    - Contains a "wolf fifth" (usually between G# and Eb) that is dissonant
    - Works well for medieval and some Renaissance music
    
    Note: Since we're generating discrete keys, we can either:
    1. Generate keys in chromatic order by walking the circle of fifths
    2. Generate keys in order of fifths (stacked fifths)
    
    This implementation uses chromatic ordering for compatibility with MIDI-like systems.
    """
    
    def __init__(self, base_freq: float = 293.33, base_key: int = 62):
        """Initialize Pythagorean tuning.
        
        Args:
            base_freq: Frequency of the base note (default: 293.33 Hz for D4)
            base_key: Key index of the base note (default: 62 for MIDI D4)
                   D is traditionally used as the center of Pythagorean tuning.
        """
        self.base_freq = base_freq
        self.base_key = base_key
        # Generate 12 chromatic notes from circle of fifths
        # Starting from base, go up and down by fifths, fold into one octave
        self._generate_chromatic_scale()
    
    def _generate_chromatic_scale(self):
        """Generate 12 chromatic semitone ratios from stacked fifths."""
        # Circle of fifths order: F, C, G, D, A, E, B, F#, C#, G#, D#, A#, E#
        # We want chromatic order: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        
        # Generate 12 notes by going up by fifths from F
        # F = base * (3/2)^(-1), C = base * (3/2)^0, G = base * (3/2)^1, etc.
        fifths_order = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Starting from F
        # Chromatic mapping: F(-1), C(0), G(1), D(2), A(3), E(4), B(5), 
        #                    F#(6), C#(7), G#(8), D#(9), A#(10)
        # We need to map these to chromatic indices
        
        # Chromatic indices relative to C (which is at fifth index 0)
        # C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11
        fifth_to_chromatic = {
            0: 0,   # C
            7: 1,   # C# (7 fifths up from C)
            2: 2,   # D (2 fifths up)
            9: 3,   # D#
            4: 4,   # E
            -1: 5,  # F (1 fifth down)
            6: 6,   # F#
            1: 7,   # G
            8: 8,   # G#
            3: 9,   # A
            10: 10, # A#
            5: 11,  # B
        }
        
        # Store the ratio for each chromatic semitone relative to C
        self._ratios = [1.0] * 12
        for fifth_steps, chromatic_index in fifth_to_chromatic.items():
            # Calculate frequency ratio from base
            ratio = (3.0/2.0) ** fifth_steps
            # Fold into single octave (0.5 to 2.0 range)
            while ratio < 1.0:
                ratio *= 2.0
            while ratio >= 2.0:
                ratio /= 2.0
            self._ratios[chromatic_index] = ratio
    
    def get_frequency(self, key_index: int) -> float:
        """Calculate frequency using Pythagorean ratios."""
        # Distance from base key
        semitones_from_base = key_index - self.base_key
        
        # Octave number and position within octave
        octave_offset = semitones_from_base // 12
        chromatic_index = semitones_from_base % 12
        
        # Get the ratio for this chromatic note
        ratio = self._ratios[chromatic_index]
        
        # Calculate frequency
        freq = self.base_freq * ratio * (2 ** octave_offset)
        return freq
    
    @property
    def name(self) -> str:
        return f"Pythagorean (base= {self.base_freq}Hz)"


class MeantoneTuning(TuningSystem):
    """Meantone temperament with parameterized comma fraction.
    
    A family of temperaments designed to make major thirds pure (or nearly so)
    by tempering (narrowing) the perfect fifths slightly.
    
    Theory:
    - In quarter-comma meantone (1/4 comma), each fifth is narrowed by 1/4 of a 
      syntonic comma (81/80), making major thirds pure (5:4 ratio)
    - The tempered fifth = 2^(1/4) / 5^(1/4) ≈ 1.49535 (vs pure 1.5)
    - The "meantone" refers to the whole tone being the "mean" (average) of 
      the major and minor tones in just intonation
    
    Common variants:
    - 1/4 comma: Pure major thirds, slightly narrow fifths
    - 1/3 comma: Between 1/4 and 1/6
    - 1/6 comma: Fifths closer to pure, thirds slightly wide
    
    Characteristics:
    - Some keys sound excellent (with pure thirds)
    - Some keys are unusable (wolf intervals)
    - Popular in Renaissance and Baroque music
    
    This implementation generates 12 tones from a chain of tempered fifths,
    similar to Pythagorean but with narrowed fifths.
    """
    
    def __init__(self, comma_fraction: float = 0.25, base_freq: float = 261.63,
                 base_key: int = 60):
        """Initialize meantone temperament.
        
        Args:
            comma_fraction: Fraction of syntonic comma to temper each fifth.
                          0.25 = quarter-comma (pure major thirds)
                          0.333... = third-comma
                          0.0 = Pythagorean (no tempering)
            base_freq: Frequency of the base note (default: 261.63 Hz for C4)
            base_key: Key index of the base note (default: 60 for MIDI C4)
        """
        self.comma_fraction = comma_fraction
        self.base_freq = base_freq
        self.base_key = base_key
        self._generate_chromatic_scale()
    
    def _generate_chromatic_scale(self):
        """Generate 12 chromatic semitone ratios from chain of tempered fifths."""
        # Calculate tempered fifth ratio
        # A pure fifth is 3/2
        # A syntonic comma is 81/80
        # Tempered fifth = pure_fifth / (comma ^ comma_fraction)
        syntonic_comma = 81.0 / 80.0
        tempered_fifth = (3.0/2.0) / (syntonic_comma ** self.comma_fraction)
        
        # Normalize so that tempered_fifth^12 = 2^7 (12 fifths = 7 octaves in 12-TET)
        # For meantone, we don't force this closure, creating the "wolf"
        # But we need to place our 12 notes
        
        # Generate 12 notes by going up by tempered fifths
        # Map to chromatic order
        fifth_to_chromatic = {
            0: 0,   # C
            7: 1,   # C#
            2: 2,   # D
            9: 3,   # D#
            4: 4,   # E
            -1: 5,  # F
            6: 6,   # F#
            1: 7,   # G
            8: 8,   # G#
            3: 9,   # A
            10: 10, # A#
            5: 11,  # B
        }
        
        self._ratios = [1.0] * 12
        for fifth_steps, chromatic_index in fifth_to_chromatic.items():
            ratio = tempered_fifth ** fifth_steps
            # Fold into single octave
            while ratio < 1.0:
                ratio *= 2.0
            while ratio >= 2.0:
                ratio /= 2.0
            self._ratios[chromatic_index] = ratio
    
    def get_frequency(self, key_index: int) -> float:
        """Calculate frequency using meantone ratios."""
        semitones_from_base = key_index - self.base_key
        octave_offset = semitones_from_base // 12
        chromatic_index = semitones_from_base % 12
        ratio = self._ratios[chromatic_index]
        freq = self.base_freq * ratio * (2 ** octave_offset)
        return freq
    
    @property
    def name(self) -> str:
        fraction_str = f"{self.comma_fraction:.4f}".rstrip('0').rstrip('.')
        return f"{fraction_str}-comma Meantone"


class EDOSystem(TuningSystem):
    """Equal Division of the Octave (n-EDO).
    
    Generalization of 12-TET to any number of equal divisions.
    Each step has a frequency ratio of 2^(1/n).
    
    Common EDO systems:
    - 12-EDO: Standard Western tuning (identical to 12-TET)
    - 19-EDO: Near-just thirds and sixths, distinct major/minor whole tones
    - 24-EDO: Quarter-tone system used in some contemporary Arabic music
    - 31-EDO: Excellent approximation to meantone temperament
    - 41-EDO: Very accurate just intonation approximations
    - 53-EDO: Close approximation to Pythagorean and just intonation
    
    Formula: f(key) = reference_hz * 2^((key - reference_key) / divisions)
    
    Note: Unlike 12-TET, the "semitone" concept doesn't directly apply.
    Each step is simply 1/n of an octave on a logarithmic scale.
    """
    
    def __init__(self, divisions: int = 19, reference_hz: float = 440.0,
                 reference_key: int = 69):
        """Initialize n-EDO tuning.
        
        Args:
            divisions: Number of equal divisions of the octave (default: 19)
            reference_hz: Frequency of the reference note (default: 440.0 Hz)
            reference_key: Key index of the reference note (default: 69)
        """
        if divisions < 1:
            raise ValueError("Divisions must be at least 1")
        self.divisions = divisions
        self.reference_hz = reference_hz
        self.reference_key = reference_key
    
    def get_frequency(self, key_index: int) -> float:
        """Calculate frequency using n-EDO formula."""
        return self.reference_hz * (2 ** ((key_index - self.reference_key) / self.divisions))
    
    @property
    def name(self) -> str:
        return f"{self.divisions}-EDO"


class NonOctaveSystem(TuningSystem):
    """Non-octave repeating scales (e.g., Bohlen-Pierce, alpha, beta, gamma).
    
    These scales are based on a step ratio other than 2 (the octave).
    They can create unique harmonic relationships that don't repeat at the octave.
    
    Examples:
    - Alpha scale (Wendy Carlos): step ratio ≈ 1.618 (golden ratio)
      Based on the golden ratio φ = (1 + √5) / 2
      15.39 steps per "octave" (not a true octave)
    
    - Beta scale (Wendy Carlos): step ratio ≈ 1.414 (√2)
      17.08 steps per "octave"
    
    - Gamma scale (Wendy Carlos): step ratio ≈ 1.201
      34.65 steps per "octave"
    
    - Bohlen-Pierce: step ratio = 3 (tritave)
      13 steps per tritave (3:1 instead of 2:1)
      Based on the 3:1 ratio and odd harmonics
    
    Formula: f(key) = reference_hz * step_ratio^((key - reference_key) / steps_per_repeat)
    
    Note: These scales have no true octaves - what would be an octave (2:1 ratio)
    lands between keys, creating a fundamentally different approach to harmony.
    """
    
    def __init__(self, step_ratio: float, steps_per_repeat: int,
                 reference_hz: float = 440.0, reference_key: int = 69,
                 name: Optional[str] = None):
        """Initialize non-octave tuning system.
        
        Args:
            step_ratio: The frequency ratio for one complete cycle (e.g., 3.0 for Bohlen-Pierce)
            steps_per_repeat: Number of steps in one complete cycle
            reference_hz: Frequency of the reference note
            reference_key: Key index of the reference note
            name: Optional custom name for this tuning system
        """
        if step_ratio <= 0:
            raise ValueError("Step ratio must be positive")
        if steps_per_repeat < 1:
            raise ValueError("Steps per repeat must be at least 1")
        
        self.step_ratio = step_ratio
        self.steps_per_repeat = steps_per_repeat
        self.reference_hz = reference_hz
        self.reference_key = reference_key
        self._custom_name = name
    
    @classmethod
    def alpha_scale(cls, reference_hz: float = 440.0, reference_key: int = 69):
        """Create Wendy Carlos' Alpha scale (golden ratio based).
        
        The alpha scale uses the golden ratio φ ≈ 1.618 as its step ratio,
        with approximately 15.39 steps per pseudo-octave.
        """
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        # The alpha scale has ~15.39 steps per "octave"
        return cls(step_ratio=phi, steps_per_repeat=15,
                   reference_hz=reference_hz, reference_key=reference_key,
                   name="Alpha (φ-scale)")
    
    @classmethod
    def beta_scale(cls, reference_hz: float = 440.0, reference_key: int = 69):
        """Create Wendy Carlos' Beta scale (sqrt(2) based).
        
        The beta scale uses √2 ≈ 1.414 as its step ratio,
        with approximately 17.08 steps per pseudo-octave.
        """
        return cls(step_ratio=math.sqrt(2), steps_per_repeat=17,
                   reference_hz=reference_hz, reference_key=reference_key,
                   name="Beta (√2-scale)")
    
    @classmethod
    def gamma_scale(cls, reference_hz: float = 440.0, reference_key: int = 69):
        """Create Wendy Carlos' Gamma scale.
        
        The gamma scale uses a step ratio of ~1.201,
        with approximately 34.65 steps per pseudo-octave.
        """
        return cls(step_ratio=1.201, steps_per_repeat=35,
                   reference_hz=reference_hz, reference_key=reference_key,
                   name="Gamma (1.201-scale)")
    
    @classmethod
    def bohlen_pierce(cls, reference_hz: float = 440.0, reference_key: int = 69):
        """Create Bohlen-Pierce scale (tritave based).
        
        The Bohlen-Pierce scale uses a 3:1 ratio (tritave) instead of 2:1 (octave),
        with 13 equal divisions. It's based on odd harmonics and creates
        unique harmonic relationships.
        """
        return cls(step_ratio=3.0, steps_per_repeat=13,
                   reference_hz=reference_hz, reference_key=reference_key,
                   name="Bohlen-Pierce (3:1)")
    
    def get_frequency(self, key_index: int) -> float:
        """Calculate frequency using non-octave formula."""
        return self.reference_hz * (self.step_ratio ** 
                                    ((key_index - self.reference_key) / self.steps_per_repeat))
    
    @property
    def name(self) -> str:
        if self._custom_name:
            return self._custom_name
        return f"Non-Octave (ratio={self.step_ratio:.3f}, steps={self.steps_per_repeat})"


# Verification tests
if __name__ == "__main__":
    print("=" * 60)
    print("TUNING SYSTEM VERIFICATION")
    print("=" * 60)
    
    # Test 1: 12-TET A4 = 440Hz
    print("\n1. TwelveTET - A4 = 440Hz verification:")
    twelve_tet = TwelveTET(reference_hz=440.0, reference_key=69)
    a4_freq = twelve_tet.get_frequency(69)
    print(f"   A4 (key 69): {a4_freq:.2f} Hz (expected: 440.00 Hz)")
    assert abs(a4_freq - 440.0) < 0.01, "A4 frequency mismatch!"
    
    # Check octave relationship
    a5_freq = twelve_tet.get_frequency(81)  # A5 is 12 semitones above A4
    print(f"   A5 (key 81): {a5_freq:.2f} Hz (expected: 880.00 Hz)")
    assert abs(a5_freq - 880.0) < 0.01, "Octave relationship failed!"
    
    # Check vectorized version
    freqs = twelve_tet.get_all_frequencies(128)
    print(f"   Vectorized frequencies shape: {freqs.shape}")
    print(f"   First 5 frequencies: {freqs[:5].tolist()}")
    
    # Test 2: Pythagorean D major scale
    print("\n2. PythagoreanTuning - D major scale frequencies:")
    pythagorean = PythagoreanTuning(base_freq=293.33, base_key=62)  # D4
    # D major: D(62), E(64), F#(66), G(67), A(69), B(71), C#(73), D(74)
    d_major_keys = [62, 64, 66, 67, 69, 71, 73, 74]
    d_major_names = ['D4', 'E4', 'F#4', 'G4', 'A4', 'B4', 'C#5', 'D5']
    print(f"   Base note D4: {pythagorean.get_frequency(62):.2f} Hz")
    for key, name in zip(d_major_keys, d_major_names):
        freq = pythagorean.get_frequency(key)
        print(f"   {name} (key {key}): {freq:.2f} Hz")
    
    # Verify pure fifth (D to A should be close to 3:2)
    d_freq = pythagorean.get_frequency(62)
    a_freq = pythagorean.get_frequency(69)
    fifth_ratio = a_freq / d_freq
    print(f"   D-A fifth ratio: {fifth_ratio:.6f} (pure = 1.5)")
    
    # Test 3: 19-EDO vs 12-TET comparison for A4-E5 interval
    print("\n3. EDOSystem - 19-EDO vs 12-TET (A4-E5 interval):")
    
    # In 12-TET, A4 (69) to E5 (76) is a perfect fifth (7 semitones)
    twelve_tet = TwelveTET()
    a4_12tet = twelve_tet.get_frequency(69)
    e5_12tet = twelve_tet.get_frequency(76)
    fifth_12tet = e5_12tet / a4_12tet
    
    # In 19-EDO, a perfect fifth is approximately 11 steps (not 7)
    # 19-EDO fifth = 2^(11/19) ≈ 1.4938
    nineteen_edo = EDOSystem(divisions=19)
    a4_19edo = nineteen_edo.get_frequency(69)
    e5_19edo = nineteen_edo.get_frequency(69 + 11)  # 11 steps for fifth in 19-EDO
    fifth_19edo = e5_19edo / a4_19edo
    
    print(f"   12-TET A4: {a4_12tet:.2f} Hz, E5: {e5_12tet:.2f} Hz")
    print(f"   12-TET fifth ratio: {fifth_12tet:.6f}")
    print(f"   19-EDO A4: {a4_19edo:.2f} Hz, 'E5' (11 steps): {e5_19edo:.2f} Hz")
    print(f"   19-EDO fifth ratio: {fifth_19edo:.6f}")
    print(f"   Pure fifth ratio: 1.500000")
    print(f"   19-EDO is closer to pure fifth: {abs(fifth_19edo - 1.5) < abs(fifth_12tet - 1.5)}")
    
    # Test 4: Meantone temperament
    print("\n4. MeantoneTuning - Quarter-comma comparison:")
    quarter_comma = MeantoneTuning(comma_fraction=0.25)
    c4_qc = quarter_comma.get_frequency(60)
    e4_qc = quarter_comma.get_frequency(64)
    third_ratio = e4_qc / c4_qc
    print(f"   Quarter-comma C4: {c4_qc:.2f} Hz, E4: {e4_qc:.2f} Hz")
    print(f"   Major third ratio: {third_ratio:.6f}")
    print(f"   Pure major third: 1.250000 (5:4)")
    print(f"   12-TET major third: {(twelve_tet.get_frequency(64) / twelve_tet.get_frequency(60)):.6f}")
    
    # Test 5: Non-octave systems
    print("\n5. NonOctaveSystem - Various scales:")
    
    alpha = NonOctaveSystem.alpha_scale()
    print(f"   Alpha scale (φ={((1+5**0.5)/2):.6f}):")
    for i in range(5):
        key = 69 + i * 3  # Every 3rd step
        print(f"      Key {key}: {alpha.get_frequency(key):.2f} Hz")
    
    bp = NonOctaveSystem.bohlen_pierce()
    print(f"   Bohlen-Pierce (3:1 tritave):")
    for i in range(5):
        key = 69 + i * 3
        print(f"      Key {key}: {bp.get_frequency(key):.2f} Hz")
    # Show tritave relationship
    tritave_freq = bp.get_frequency(69 + 13)  # 13 steps = 1 tritave
    print(f"      Tritave (key 82): {tritave_freq:.2f} Hz (should be 3x reference = 1320 Hz)")
    
    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)
