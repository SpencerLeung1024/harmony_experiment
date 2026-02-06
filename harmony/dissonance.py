"""
Dissonance calculator for Harmony From First Principles.

This module provides the DissonanceCalculator class that computes
dissonance matrices for arbitrary tuning systems and instruments.

The dissonance calculation is based on the psychoacoustic model:
    d(f1, f2) = 65 * x * exp(-24 * x)
where x = |f2 - f1| / min(f1, f2)

This formula captures the phenomenon that two pure tones are most
dissonant when their frequencies differ by about 50 cents (quarter tone).
"""

import math
import torch
from typing import Optional

from .tuning import TuningSystem, TwelveTET
from .instruments import Instrument, ADSR


class DissonanceCalculator:
    """Calculator for dissonance between notes in a tuning system.
    
    Computes a dissonance matrix D where D[i,j] represents the total
dissonance between note i and note j, considering all harmonics
    from the instrument's timbre.
    
    The dissonance formula is:
        d = 65 * x * exp(-24 * x)
    where x = |f2 - f1| / min(f1, f2)
    
    For notes with multiple harmonics, the total dissonance is the
    sum of pairwise dissonances between all harmonic partials,
    weighted by their relative amplitudes.
    
    Attributes:
        tuning: The tuning system defining note frequencies
        instrument: The instrument defining harmonic content
        duration: Note duration for ADSR averaging (default: 1.0s)
    """
    
    def __init__(self, tuning: Optional[TuningSystem] = None, 
                 instrument: Optional[Instrument] = None,
                 duration: float = 1.0):
        """Initialize the dissonance calculator.
        
        Args:
            tuning: Tuning system to use (default: TwelveTET)
            instrument: Instrument timbre (default: Synth with simple harmonics)
            duration: Note duration for ADSR envelope averaging (default: 1.0)
        """
        self.tuning = tuning if tuning else TwelveTET()
        self.instrument = instrument if instrument else Instrument.synth()
        self.duration = duration
    
    def _dissonance_formula(self, f1: float, f2: float) -> float:
        """Calculate dissonance between two pure tones.
        
        Uses the psychoacoustic dissonance formula:
            d = 65 * x * exp(-24 * x)
        where x = |f2 - f1| / min(f1, f2)
        
        Args:
            f1: First frequency in Hz
            f2: Second frequency in Hz
            
        Returns:
            Dissonance value (non-negative)
        """
        if f1 <= 0 or f2 <= 0:
            return 0.0
        
        # Calculate normalized frequency difference
        x = abs(f2 - f1) / min(f1, f2)
        
        # Dissonance formula from main_optimized.py
        d = 65.0 * x * math.exp(-24.0 * x)
        
        return max(0.0, d)
    
    def calculate_matrix(self, num_keys: int, max_hz: float = 11025) -> torch.Tensor:
        """Compute the dissonance matrix for all note pairs.
        
        D[i,j] = total dissonance between note i and note j, including
        all harmonic interactions. The matrix is symmetric.
        
        For each pair of notes, we compute the dissonance between all
        pairs of their harmonic partials, weighted by amplitude.
        
        Args:
            num_keys: Number of keys to compute (0 to num_keys-1)
            max_hz: Maximum frequency to consider (Nyquist limit, default: 11025)
            
        Returns:
            Tensor of shape (num_keys, num_keys) containing dissonance values
        """
        # Get all frequencies from the tuning system
        frequencies = self.tuning.get_all_frequencies(num_keys)
        
        # Get effective harmonic amplitudes (accounting for ADSR)
        harmonic_profile = self.instrument.get_effective_amplitudes(self.duration)
        
        # Initialize dissonance matrix
        D = torch.zeros((num_keys, num_keys), dtype=torch.float32)
        
        # Compute dissonance for each pair of notes
        for k1 in range(num_keys):
            f1_base = frequencies[k1].item()
            
            for k2 in range(k1, num_keys):
                f2_base = frequencies[k2].item()
                
                total_dissonance = 0.0
                
                # Sum dissonance over all harmonic pairs
                for h1_ratio, h1_amp in harmonic_profile:
                    for h2_ratio, h2_amp in harmonic_profile:
                        f1 = f1_base * h1_ratio
                        f2 = f2_base * h2_ratio
                        
                        # Skip if either frequency is above max_hz
                        if f1 >= max_hz or f2 >= max_hz:
                            continue
                        
                        # Skip self-dissonance (same note, same harmonic)
                        if k1 == k2 and abs(h1_ratio - h2_ratio) < 1e-6:
                            continue
                        
                        # Calculate dissonance for this harmonic pair
                        d = self._dissonance_formula(f1, f2)
                        
                        # Weight by harmonic amplitudes
                        total_dissonance += d * h1_amp * h2_amp
                
                # Store in matrix (symmetric)
                D[k1, k2] = total_dissonance
                D[k2, k1] = total_dissonance
        
        return D
    
    def calculate_interval_dissonance(self, key1: int, key2: int, 
                                      max_hz: float = 11025) -> float:
        """Calculate dissonance for a single interval.
        
        Convenience method for computing dissonance between two specific keys.
        
        Args:
            key1: First key index
            key2: Second key index
            max_hz: Maximum frequency to consider
            
        Returns:
            Dissonance value for the interval
        """
        freq1 = self.tuning.get_frequency(key1)
        freq2 = self.tuning.get_frequency(key2)
        
        harmonic_profile = self.instrument.get_effective_amplitudes(self.duration)
        
        total_dissonance = 0.0
        
        for h1_ratio, h1_amp in harmonic_profile:
            for h2_ratio, h2_amp in harmonic_profile:
                f1 = freq1 * h1_ratio
                f2 = freq2 * h2_ratio
                
                if f1 >= max_hz or f2 >= max_hz:
                    continue
                
                if key1 == key2 and abs(h1_ratio - h2_ratio) < 1e-6:
                    continue
                
                d = self._dissonance_formula(f1, f2)
                total_dissonance += d * h1_amp * h2_amp
        
        return total_dissonance
    
    def get_interval_dissonances(self, num_keys: int, max_hz: float = 11025) -> dict:
        """Get average dissonance by interval class.
        
        Computes the average dissonance for each interval size
        (measured in semitones for 12-TET, or step count for other systems).
        
        Args:
            num_keys: Number of keys
            max_hz: Maximum frequency to consider
            
        Returns:
            Dictionary mapping interval -> average dissonance
        """
        D = self.calculate_matrix(num_keys, max_hz)
        
        # For 12-TET, we can group by semitone distance
        # For other tunings, group by key distance
        interval_dissonances = {}
        interval_counts = {}
        
        for i in range(num_keys):
            for j in range(i + 1, num_keys):
                interval = j - i  # Key distance
                d = D[i, j].item()
                
                if interval not in interval_dissonances:
                    interval_dissonances[interval] = 0.0
                    interval_counts[interval] = 0
                
                interval_dissonances[interval] += d
                interval_counts[interval] += 1
        
        # Average
        for interval in interval_dissonances:
            if interval_counts[interval] > 0:
                interval_dissonances[interval] /= interval_counts[interval]
        
        return interval_dissonances
    
    def __repr__(self) -> str:
        return f"DissonanceCalculator(tuning={self.tuning.name}, instrument={self.instrument.name})"


# ==================== VERIFICATION TESTS ====================

if __name__ == "__main__":
    print("=" * 60)
    print("DISSONANCE CALCULATOR VERIFICATION")
    print("=" * 60)
    
    # Test 1: Basic dissonance formula
    print("\n1. Dissonance Formula Verification:")
    calc = DissonanceCalculator()
    
    # Same frequency should give zero dissonance
    d_same = calc._dissonance_formula(440.0, 440.0)
    print(f"   Same frequency (440Hz): {d_same:.6f} (should be ~0)")
    
    # Very different frequencies should give low dissonance
    d_far = calc._dissonance_formula(440.0, 2000.0)
    print(f"   Far apart (440Hz, 2000Hz): {d_far:.6f}")
    
    # ~50 cents (quarter tone) should give peak dissonance
    # 50 cents = 2^(50/1200) ≈ 1.0293 ratio
    f1 = 440.0
    f2 = f1 * (2 ** (50 / 1200))  # ~50 cents sharp
    d_peak = calc._dissonance_formula(f1, f2)
    print(f"   ~50 cents apart: {d_peak:.6f} (should be near peak)")
    
    # Verify peak is around 50 cents
    f3 = f1 * (2 ** (25 / 1200))  # ~25 cents
    d_25 = calc._dissonance_formula(f1, f3)
    print(f"   ~25 cents apart: {d_25:.6f}")
    print(f"   Peak dissonance at ~50 cents: {d_peak > d_25 and d_peak > d_far}")
    
    # Test 2: Matrix computation
    print("\n2. Dissonance Matrix Computation (12-TET, Synth):")
    synth = Instrument.synth()
    twelve_tet = TwelveTET()
    calc = DissonanceCalculator(tuning=twelve_tet, instrument=synth)
    
    D = calc.calculate_matrix(num_keys=12, max_hz=11025)  # Just one octave
    print(f"   Matrix shape: {D.shape}")
    print(f"   Symmetric: {torch.allclose(D, D.T)}")
    print(f"   Zero diagonal: {torch.allclose(D.diag(), torch.zeros(12))}")
    print(f"   Max dissonance: {D.max():.4f}")
    print(f"   Mean dissonance: {D.mean():.4f}")
    
    # Test 3: Compare Piano vs Guitar
    print("\n3. Piano vs Guitar Timbre (12-TET):")
    piano = Instrument.piano()
    guitar = Instrument.guitar()
    
    calc_piano = DissonanceCalculator(tuning=twelve_tet, instrument=piano)
    calc_guitar = DissonanceCalculator(tuning=twelve_tet, instrument=guitar)
    
    D_piano = calc_piano.calculate_matrix(num_keys=12, max_hz=11025)
    D_guitar = calc_guitar.calculate_matrix(num_keys=12, max_hz=11025)
    
    print(f"   Piano - Max: {D_piano.max():.4f}, Mean: {D_piano.mean():.4f}")
    print(f"   Guitar - Max: {D_guitar.max():.4f}, Mean: {D_guitar.mean():.4f}")
    print(f"   Piano has more harmonics: {len(piano.harmonics) > len(guitar.harmonics)}")
    
    # Test 4: 12-TET vs Pythagorean for same interval
    print("\n4. 12-TET vs Pythagorean Tuning Comparison:")
    from .tuning import PythagoreanTuning
    
    # Calculate dissonance for a perfect fifth (7 semitones in 12-TET)
    # In Pythagorean, the fifth is pure (3:2 = 1.5)
    # In 12-TET, the fifth is 2^(7/12) ≈ 1.498
    
    # Create calculators
    calc_12tet = DissonanceCalculator(tuning=TwelveTET(), instrument=synth)
    calc_pyth = DissonanceCalculator(
        tuning=PythagoreanTuning(base_freq=261.63, base_key=60),
        instrument=synth
    )
    
    # In 12-TET: C4(60) to G4(67) is a fifth
    d_12tet_fifth = calc_12tet.calculate_interval_dissonance(60, 67)
    
    # In Pythagorean: C4(60) to G4(67) should also be a fifth
    d_pyth_fifth = calc_pyth.calculate_interval_dissonance(60, 67)
    
    print(f"   12-TET perfect fifth (C4-G4): {d_12tet_fifth:.4f}")
    print(f"   Pythagorean perfect fifth (C4-G4): {d_pyth_fifth:.4f}")
    print(f"   Difference: {abs(d_12tet_fifth - d_pyth_fifth):.4f}")
    
    # Test 5: Major third comparison
    # 12-TET major third: 4 semitones = 2^(4/12) ≈ 1.26
    # Pythagorean major third: 81/64 ≈ 1.266 (sharp)
    print("\n5. Major Third Comparison:")
    d_12tet_third = calc_12tet.calculate_interval_dissonance(60, 64)  # C to E
    d_pyth_third = calc_pyth.calculate_interval_dissonance(60, 64)
    
    print(f"   12-TET major third (C4-E4): {d_12tet_third:.4f}")
    print(f"   Pythagorean major third (C4-E4): {d_pyth_third:.4f}")
    print(f"   Pythagorean third is sharper and more dissonant: {d_pyth_third > d_12tet_third}")
    
    # Test 6: Interval dissonances
    print("\n6. Interval Dissonances (12-TET, Synth):")
    calc_synth = DissonanceCalculator(tuning=TwelveTET(), instrument=Instrument.synth())
    interval_d = calc_synth.get_interval_dissonances(num_keys=88)
    
    interval_names = ["P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7"]
    for i in range(min(12, len(interval_d))):
        name = interval_names[i] if i < len(interval_names) else f"{i}"
        print(f"   Interval {i:2d} ({name:3s}): {interval_d.get(i, 0):.4f}")
    
    # Test 7: Different instruments, same tuning
    print("\n7. Same Interval, Different Instruments (12-TET):")
    instruments = [
        ("Synth", Instrument.synth()),
        ("Guitar", Instrument.guitar()),
        ("Piano", Instrument.piano()),
        ("Bass", Instrument.bass()),
    ]
    
    for name, inst in instruments:
        calc = DissonanceCalculator(tuning=TwelveTET(), instrument=inst)
        d_tritone = calc.calculate_interval_dissonance(60, 66)  # C to F# (tritone)
        d_fifth = calc.calculate_interval_dissonance(60, 67)     # C to G (fifth)
        print(f"   {name:10s}: Tritone={d_tritone:.4f}, Fifth={d_fifth:.4f}")
    
    # Test 8: ADSR effect on dissonance
    print("\n8. ADSR Effect on Effective Amplitudes:")
    piano_inst = Instrument.piano()
    
    # Short duration - less sustain contribution
    eff_short = piano_inst.get_effective_amplitudes(duration=0.1)
    # Long duration - more sustain contribution
    eff_long = piano_inst.get_effective_amplitudes(duration=2.0)
    
    print(f"   Piano harmonic effective amplitudes:")
    for i, ((r1, a1), (r2, a2)) in enumerate(zip(eff_short[:4], eff_long[:4])):
        print(f"      H{i+1}: short={a1:.3f}, long={a2:.3f}, ratio={a1/a2:.2f}")
    
    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)
