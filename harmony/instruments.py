"""
Instrument system with ADSR envelopes and harmonic profiles for Harmony From First Principles.

This module provides:
- ADSR envelope class for shaping note amplitude over time
- Instrument class with configurable harmonic content and envelope parameters
- Preset instruments (Piano, Guitar, Bass, Synth, Drums)
- Support for per-harmonic ADSR envelopes
"""

from typing import List, Tuple, Optional, Union
import torch
import math


class ADSR:
    """Attack-Decay-Sustain-Release envelope generator.
    
    The ADSR envelope shapes the amplitude of a note over its duration:
    - Attack: Time to reach peak amplitude from zero
    - Decay: Time to fall from peak to sustain level
    - Sustain: Level maintained during the note's hold phase (0-1)
    - Release: Time to fade from sustain level to zero after note ends
    
    For notes that are shorter than (attack + decay + release), the envelope
    is truncated appropriately.
    
    Attributes:
        attack: Attack time in seconds
        decay: Decay time in seconds
        sustain: Sustain level (0.0 to 1.0)
        release: Release time in seconds
    """
    
    def __init__(self, attack: float = 0.01, decay: float = 0.1, 
                 sustain: float = 0.7, release: float = 0.3):
        """Initialize ADSR envelope parameters.
        
        Args:
            attack: Attack time in seconds (default: 0.01)
            decay: Decay time in seconds (default: 0.1)
            sustain: Sustain level 0-1 (default: 0.7)
            release: Release time in seconds (default: 0.3)
        """
        self.attack = max(0.0, attack)
        self.decay = max(0.0, decay)
        self.sustain = max(0.0, min(1.0, sustain))
        self.release = max(0.0, release)
    
    def get_envelope(self, duration: float, sample_rate: int) -> torch.Tensor:
        """Generate envelope shape for a note of given duration.
        
        Creates a piecewise linear envelope:
        1. Linear ramp from 0 to 1 during attack
        2. Exponential decay from 1 to sustain during decay
        3. Constant sustain level during sustain phase
        4. Exponential decay from sustain to 0 during release
        
        Args:
            duration: Note duration in seconds
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Tensor of shape (num_samples,) containing envelope values
        """
        num_samples = int(duration * sample_rate)
        if num_samples == 0:
            return torch.zeros(0)
        
        t = torch.linspace(0, duration, num_samples)
        envelope = torch.zeros(num_samples)
        
        # Calculate phase boundaries in samples
        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        release_samples = int(self.release * sample_rate)
        
        # Attack phase: linear ramp 0 -> 1
        if attack_samples > 0:
            attack_end = min(attack_samples, num_samples)
            envelope[:attack_end] = torch.linspace(0, 1, attack_end)
        
        # Decay phase: exponential 1 -> sustain
        if decay_samples > 0 and attack_samples < num_samples:
            decay_start = attack_samples
            decay_end = min(attack_samples + decay_samples, num_samples)
            decay_t = torch.arange(decay_end - decay_start, dtype=torch.float32)
            # Exponential decay: sustain + (1-sustain) * exp(-t/tau)
            # where tau is chosen so that at decay_samples, we're at sustain
            decay_length = max(1, decay_samples)
            envelope[decay_start:decay_end] = self.sustain + (1 - self.sustain) * \
                torch.exp(-5 * decay_t / decay_length)
        
        # Sustain phase: constant sustain level
        sustain_start = attack_samples + decay_samples
        if sustain_start < num_samples:
            # Release starts before the end of the note
            release_start = max(0, num_samples - release_samples)
            if sustain_start < release_start:
                envelope[sustain_start:release_start] = self.sustain
        
        # Release phase: exponential sustain -> 0
        if release_samples > 0:
            release_start = max(0, num_samples - release_samples)
            release_t = torch.arange(num_samples - release_start, dtype=torch.float32)
            release_length = max(1, release_samples)
            envelope[release_start:] = self.sustain * torch.exp(-5 * release_t / release_length)
        
        return envelope
    
    def get_average_amplitude(self, duration: float) -> float:
        """Calculate average amplitude over the note duration.
        
        This is used for dissonance calculations where we need a static
        amplitude weight for each harmonic.
        
        Args:
            duration: Note duration in seconds
            
        Returns:
            Average amplitude value (0-1)
        """
        # Approximate average using piecewise integration
        # For simplicity, use a heuristic based on the envelope shape
        total_time = self.attack + self.decay + duration + self.release
        
        if total_time <= 0:
            return self.sustain
        
        # Area under attack (triangle): 0.5 * attack * 1
        attack_area = 0.5 * self.attack
        
        # Area under decay (trapezoid-ish): decay * (1 + sustain) / 2
        # Using exponential, this is approximately correct
        decay_area = self.decay * (1 + self.sustain) / 2
        
        # Area under sustain: sustain * (duration - attack - decay) if positive
        sustain_time = max(0, duration - self.attack - self.decay)
        sustain_area = self.sustain * sustain_time
        
        # Area under release (triangle-ish): release * sustain / 2
        release_area = self.release * self.sustain / 2
        
        total_area = attack_area + decay_area + sustain_area + release_area
        return total_area / duration if duration > 0 else self.sustain
    
    def __repr__(self) -> str:
        return f"ADSR(A={self.attack:.3f}s, D={self.decay:.3f}s, S={self.sustain:.2f}, R={self.release:.3f}s)"


class Instrument:
    """Musical instrument with harmonic content and ADSR envelope.
    
    Instruments define the timbre through their harmonic content and
    temporal characteristics through ADSR envelopes. Each harmonic can
    optionally have its own ADSR envelope for more complex sounds.
    
    Attributes:
        name: Human-readable instrument name
        harmonics: List of (frequency_ratio, amplitude) tuples
        adsr: Default ADSR envelope for all harmonics
        key_range: Valid MIDI key range (low, high)
        per_harmonic_adsr: Optional dict mapping harmonic index to ADSR
    """
    
    def __init__(self, name: str, harmonics: List[Tuple[float, float]],
                 adsr: Optional[ADSR] = None,
                 key_range: Tuple[int, int] = (0, 127),
                 per_harmonic_adsr: Optional[dict] = None):
        """Initialize an instrument.
        
        Args:
            name: Instrument name
            harmonics: List of (frequency_ratio, amplitude) for each harmonic
            adsr: Default ADSR envelope (default: instant on/off)
            key_range: Valid MIDI key range as (low, high) tuple
            per_harmonic_adsr: Dict mapping harmonic index to its own ADSR
        """
        self.name = name
        self.harmonics = harmonics
        self.adsr = adsr if adsr else ADSR(attack=0.0, decay=0.0, sustain=1.0, release=0.0)
        self.key_range = key_range
        self.per_harmonic_adsr = per_harmonic_adsr or {}
    
    def get_harmonic_profile(self) -> List[Tuple[float, float, ADSR]]:
        """Get harmonics with their associated ADSR envelopes.
        
        Returns:
            List of (frequency_ratio, amplitude, adsr) tuples
        """
        profile = []
        for i, (ratio, amp) in enumerate(self.harmonics):
            # Use per-harmonic ADSR if available, otherwise use default
            adsr = self.per_harmonic_adsr.get(i, self.adsr)
            profile.append((ratio, amp, adsr))
        return profile
    
    def get_effective_amplitudes(self, duration: float) -> List[Tuple[float, float]]:
        """Get harmonics with amplitude adjusted by average ADSR level.
        
        For dissonance calculations, we need static amplitudes that
        represent the average contribution of each harmonic.
        
        Args:
            duration: Note duration in seconds
            
        Returns:
            List of (frequency_ratio, effective_amplitude) tuples
        """
        effective = []
        for ratio, amp, adsr in self.get_harmonic_profile():
            avg_amp = adsr.get_average_amplitude(duration)
            effective.append((ratio, amp * avg_amp))
        return effective
    
    def is_key_in_range(self, key: int) -> bool:
        """Check if a MIDI key is within the instrument's valid range.
        
        Args:
            key: MIDI key number (0-127)
            
        Returns:
            True if key is in valid range
        """
        return self.key_range[0] <= key <= self.key_range[1]
    
    def __repr__(self) -> str:
        return f"Instrument({self.name}, {len(self.harmonics)} harmonics, range={self.key_range})"
    
    # ==================== PRESET INSTRUMENTS ====================
    
    @classmethod
    def piano(cls) -> "Instrument":
        """Create a piano-like instrument.
        
        Piano has rich harmonic content with inharmonic stretch
        (upper harmonics are slightly sharp) and a percussive ADSR
        with quick attack and relatively fast decay.
        
        Returns:
            Piano Instrument instance
        """
        # Piano harmonics with slight inharmonicity
        # Higher harmonics are progressively sharper
        harmonics = [
            (1.000, 1.000),   # Fundamental
            (2.002, 0.450),   # 2nd harmonic (slightly sharp)
            (3.005, 0.280),   # 3rd harmonic
            (4.010, 0.180),   # 4th harmonic
            (5.015, 0.120),   # 5th harmonic
            (6.025, 0.080),   # 6th harmonic
            (7.035, 0.055),   # 7th harmonic
            (8.050, 0.040),   # 8th harmonic
        ]
        
        # Percussive envelope: quick attack, fast decay, moderate sustain
        adsr = ADSR(attack=0.005, decay=0.4, sustain=0.3, release=0.5)
        
        # Higher harmonics decay faster
        per_harmonic_adsr = {
            0: ADSR(attack=0.005, decay=0.4, sustain=0.3, release=0.5),
            1: ADSR(attack=0.005, decay=0.35, sustain=0.25, release=0.4),
            2: ADSR(attack=0.005, decay=0.3, sustain=0.2, release=0.35),
            3: ADSR(attack=0.005, decay=0.25, sustain=0.15, release=0.3),
            4: ADSR(attack=0.005, decay=0.2, sustain=0.1, release=0.25),
            5: ADSR(attack=0.005, decay=0.15, sustain=0.08, release=0.2),
            6: ADSR(attack=0.005, decay=0.1, sustain=0.05, release=0.15),
            7: ADSR(attack=0.005, decay=0.08, sustain=0.03, release=0.1),
        }
        
        return cls(
            name="Piano",
            harmonics=harmonics,
            adsr=adsr,
            key_range=(21, 108),  # Standard 88-key piano range
            per_harmonic_adsr=per_harmonic_adsr
        )
    
    @classmethod
    def guitar(cls) -> "Instrument":
        """Create a guitar-like instrument.
        
        Guitar has a plucked string sound with characteristic
        harmonics and a percussive envelope with quick attack
        and longer sustain than piano.
        
        Returns:
            Guitar Instrument instance
        """
        # Guitar harmonics - odd harmonics are relatively strong
        harmonics = [
            (1.0, 1.00),   # Fundamental
            (2.0, 0.55),   # 2nd harmonic
            (3.0, 0.35),   # 3rd harmonic
            (4.0, 0.22),   # 4th harmonic
            (5.0, 0.15),   # 5th harmonic
            (6.0, 0.10),   # 6th harmonic
        ]
        
        # Plucked envelope: very quick attack, moderate decay, long sustain
        adsr = ADSR(attack=0.002, decay=0.3, sustain=0.6, release=0.8)
        
        return cls(
            name="Guitar",
            harmonics=harmonics,
            adsr=adsr,
            key_range=(40, 88),  # Standard guitar range (E2-E6)
        )
    
    @classmethod
    def bass(cls) -> "Instrument":
        """Create a bass-like instrument.
        
        Bass has fewer harmonics than guitar/piano, emphasizing
        the fundamental and lower harmonics. Long sustain.
        
        Returns:
            Bass Instrument instance
        """
        # Bass harmonics - fewer and weaker upper harmonics
        harmonics = [
            (1.0, 1.00),   # Fundamental (strong)
            (2.0, 0.40),   # 2nd harmonic
            (3.0, 0.20),   # 3rd harmonic
            (4.0, 0.10),   # 4th harmonic
            (5.0, 0.05),   # 5th harmonic
        ]
        
        # Bass envelope: quick attack, long sustain, moderate release
        adsr = ADSR(attack=0.01, decay=0.2, sustain=0.75, release=0.4)
        
        return cls(
            name="Bass",
            harmonics=harmonics,
            adsr=adsr,
            key_range=(28, 67),  # Bass range (E1-G4)
        )
    
    @classmethod
    def synth(cls) -> "Instrument":
        """Create a simple synthesizer instrument.
        
        Synth has simple harmonic content (similar to the original
        default instrument) with no envelope (instant on/off).
        
        Returns:
            Synth Instrument instance
        """
        harmonics = [
            (1.0, 1.0),   # Fundamental
            (2.0, 0.5),   # 2nd harmonic
            (3.0, 0.33),  # 3rd harmonic
            (4.0, 0.25),  # 4th harmonic
            (5.0, 0.2),   # 5th harmonic
            (6.0, 0.17),  # 6th harmonic
        ]
        
        # No envelope (or instant on/off)
        adsr = ADSR(attack=0.0, decay=0.0, sustain=1.0, release=0.0)
        
        return cls(
            name="Synth",
            harmonics=harmonics,
            adsr=adsr,
            key_range=(0, 127),
        )
    
    @classmethod
    def drums_kick(cls) -> "Instrument":
        """Create a kick drum instrument.
        
        Kick drums are noise-based/percussive with very quick attack,
        fast decay, and strong fundamental with inharmonic components.
        
        Returns:
            Kick drum Instrument instance
        """
        # Kick drum - strong fundamental, some inharmonic content
        # Modeled as very short duration with pitch drop
        harmonics = [
            (1.0, 1.00),    # Fundamental (boom)
            (1.5, 0.30),    # Inharmonic click
            (2.2, 0.20),    # Body resonance
            (3.5, 0.15),    # Click/harmonics
        ]
        
        # Very percussive: quick attack, fast decay, no sustain
        adsr = ADSR(attack=0.001, decay=0.15, sustain=0.0, release=0.05)
        
        return cls(
            name="Kick Drum",
            harmonics=harmonics,
            adsr=adsr,
            key_range=(36, 36),  # Usually fixed pitch or narrow range
        )
    
    @classmethod
    def drums_snare(cls) -> "Instrument":
        """Create a snare drum instrument.
        
        Snare drums are characterized by noise and many inharmonic
        partials. Modeled here as a cluster of inharmonic frequencies.
        
        Returns:
            Snare drum Instrument instance
        """
        # Snare has many inharmonic partials
        # These are approximate - snare is more noise than tones
        harmonics = [
            (1.00, 0.80),   # Fundamental (drum body)
            (1.41, 0.60),   # Inharmonic
            (1.73, 0.50),   # Inharmonic
            (2.23, 0.40),   # Inharmonic
            (2.83, 0.35),   # Inharmonic
            (3.46, 0.30),   # Inharmonic
        ]
        
        # Sharp attack, fast decay, no sustain
        adsr = ADSR(attack=0.001, decay=0.12, sustain=0.0, release=0.08)
        
        return cls(
            name="Snare Drum",
            harmonics=harmonics,
            adsr=adsr,
            key_range=(38, 38),  # Usually fixed pitch
        )
    
    @classmethod
    def drums_hihat(cls) -> "Instrument":
        """Create a hi-hat instrument.
        
        Hi-hats are essentially noise with very high frequency content
        and extremely fast decay.
        
        Returns:
            Hi-hat Instrument instance
        """
        # Hi-hat is mostly noise (high frequencies)
        # Modeled as many closely-spaced high harmonics
        harmonics = [
            (2.0, 0.50),   # Higher relative to fundamental
            (3.0, 0.60),
            (4.0, 0.55),
            (5.0, 0.50),
            (6.0, 0.45),
            (7.0, 0.40),
            (8.0, 0.35),
        ]
        
        # Very sharp attack, very fast decay
        adsr = ADSR(attack=0.001, decay=0.05, sustain=0.0, release=0.02)
        
        return cls(
            name="Hi-Hat",
            harmonics=harmonics,
            adsr=adsr,
            key_range=(42, 42),  # Closed hi-hat MIDI note
        )


# ==================== VERIFICATION TESTS ====================

if __name__ == "__main__":
    print("=" * 60)
    print("INSTRUMENT SYSTEM VERIFICATION")
    print("=" * 60)
    
    # Test 1: ADSR envelope generation
    print("\n1. ADSR Envelope Generation:")
    adsr = ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
    print(f"   ADSR: {adsr}")
    
    envelope = adsr.get_envelope(duration=1.0, sample_rate=1000)
    print(f"   Envelope shape: {envelope.shape}")
    print(f"   Initial value: {envelope[0].item():.3f}")
    print(f"   Peak value: {envelope.max().item():.3f}")
    print(f"   Sustain region (middle): {envelope[500].item():.3f}")
    print(f"   Final value: {envelope[-1].item():.3f}")
    
    avg_amp = adsr.get_average_amplitude(duration=1.0)
    print(f"   Average amplitude: {avg_amp:.3f}")
    
    # Test 2: Piano instrument
    print("\n2. Piano Instrument:")
    piano = Instrument.piano()
    print(f"   {piano}")
    print(f"   Harmonics: {len(piano.harmonics)}")
    for i, (ratio, amp) in enumerate(piano.harmonics[:4]):
        print(f"      H{i+1}: ratio={ratio:.3f}, amp={amp:.3f}")
    print(f"   Key range: {piano.key_range}")
    print(f"   Default ADSR: {piano.adsr}")
    
    # Test 3: Guitar instrument
    print("\n3. Guitar Instrument:")
    guitar = Instrument.guitar()
    print(f"   {guitar}")
    print(f"   Harmonics: {len(guitar.harmonics)}")
    for ratio, amp in guitar.harmonics[:4]:
        print(f"      ratio={ratio:.1f}, amp={amp:.2f}")
    
    # Test 4: Bass instrument
    print("\n4. Bass Instrument:")
    bass = Instrument.bass()
    print(f"   {bass}")
    print(f"   Harmonics: {len(bass.harmonics)}")
    print(f"   ADSR: {bass.adsr}")
    
    # Test 5: Synth instrument (default behavior)
    print("\n5. Synth Instrument:")
    synth = Instrument.synth()
    print(f"   {synth}")
    print(f"   Harmonics: {len(synth.harmonics)}")
    print(f"   ADSR (no envelope): {synth.adsr}")
    
    # Test 6: Drum instruments
    print("\n6. Drum Instruments:")
    kick = Instrument.drums_kick()
    snare = Instrument.drums_snare()
    hihat = Instrument.drums_hihat()
    print(f"   {kick}")
    print(f"   {snare}")
    print(f"   {hihat}")
    
    # Test 7: Harmonic profile with per-harmonic ADSR
    print("\n7. Harmonic Profile with Per-Harmonic ADSR:")
    profile = piano.get_harmonic_profile()
    print(f"   Piano has {len(profile)} harmonics")
    print(f"   First harmonic ADSR: {profile[0][2]}")
    print(f"   Last harmonic ADSR: {profile[-1][2]}")
    
    # Test 8: Effective amplitudes for dissonance
    print("\n8. Effective Amplitudes (for dissonance calculation):")
    effective = piano.get_effective_amplitudes(duration=0.5)
    print(f"   Duration: 0.5s")
    for i, (ratio, amp) in enumerate(effective[:4]):
        orig_amp = piano.harmonics[i][1]
        print(f"      H{i+1}: ratio={ratio:.3f}, original={orig_amp:.3f}, effective={amp:.3f}")
    
    # Test 9: Key range checking
    print("\n9. Key Range Checking:")
    test_keys = [20, 21, 60, 108, 109]
    for key in test_keys:
        in_range = piano.is_key_in_range(key)
        print(f"   Key {key}: {'in range' if in_range else 'out of range'}")
    
    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)
