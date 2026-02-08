from typing import Optional, Union, Tuple, List, Dict, Callable
from functools import lru_cache
import numpy as np
import torch

# Module-level cached functions for static caching
@lru_cache(maxsize=1024)
def _cached_get_envelope(attack: float, decay: float, sustain: float, release: float, duration: float, sample_rate: int) -> np.ndarray:
    """Static cached envelope generation."""
    attack_start = 0
    decay_start = int(attack * sample_rate)
    sustain_start = decay_start + int(decay * sample_rate)
    release_start = int(duration * sample_rate)
    note_end = release_start + int(release * sample_rate)

    # Check the case where the note duration is shorter than the attack + decay time
    if sustain_start > release_start:
        sustain_start = release_start

    envelope = np.zeros(note_end)

    # Attack
    if decay_start > attack_start:
        envelope[attack_start:decay_start] = np.linspace(0, 1, decay_start - attack_start)

    # Decay
    if sustain_start > decay_start:
        envelope[decay_start:sustain_start] = np.linspace(1, sustain, sustain_start - decay_start)

    # Sustain
    if sustain_start < release_start:
        envelope[sustain_start:release_start] = sustain

    # Release
    if note_end > release_start:
        envelope[release_start:note_end] = np.linspace(sustain, 0, note_end - release_start)

    return envelope

@lru_cache(maxsize=1024)
def _cached_mean_amplitude(attack: float, decay: float, sustain: float, release: float, duration: float, sample_rate: int) -> float:
    """Static cached mean amplitude calculation."""
    envelope = _cached_get_envelope(attack, decay, sustain, release, duration, sample_rate)
    return np.mean(envelope)

class ADSR:
    def __init__(
        self,
        attack: float,
        decay: float,
        sustain: float,
        release: float
    ):
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
    
    def get_envelope(self, duration: float, sample_rate: int) -> np.ndarray:
        """Get envelope using static cache."""
        return _cached_get_envelope(self.attack, self.decay, self.sustain, self.release, duration, sample_rate)
        attack_start = 0
        decay_start = int(self.attack * sample_rate)
        sustain_start = decay_start + int(self.decay * sample_rate)
        release_start = int(duration * sample_rate)
        note_end = release_start + int(self.release * sample_rate)

        # Check the case where the note duration is shorter than the attack + decay time
        if sustain_start > release_start:
            sustain_start = release_start

        envelope = np.zeros(note_end)

        # Attack
        envelope[attack_start:decay_start] = np.linspace(0, 1, decay_start - attack_start)

        # Decay
        envelope[decay_start:sustain_start] = np.linspace(1, self.sustain, sustain_start - decay_start)

        # Sustain
        if sustain_start < release_start:
            envelope[sustain_start:release_start] = self.sustain

        # Release
        envelope[release_start:note_end] = np.linspace(self.sustain, 0, note_end - release_start)

        return envelope
    
    def mean_amplitude(self, duration: float, sample_rate: int) -> float:
        """Get mean amplitude using static cache."""
        return _cached_mean_amplitude(self.attack, self.decay, self.sustain, self.release, duration, sample_rate)

# Module-level cached sound generation
@lru_cache(maxsize=2048)
def _cached_get_sound(
    harmonics_tuple: Tuple[Tuple[float, float], ...],
    adsr_attack: float,
    adsr_decay: float,
    adsr_sustain: float,
    adsr_release: float,
    harmonic_adsrs_tuple: Tuple[Tuple[int, float, float, float, float], ...],
    freq: float,
    velocity: float,
    duration: float,
    sample_rate: int
) -> np.ndarray:
    """Static cached sound generation.
    
    Note: harmonics and harmonic_adsrs are converted to tuples for hashability.
    """
    # Find max release time
    max_release = max([adsr_release] + [h[4] for h in harmonic_adsrs_tuple])
    total_duration = duration + max_release
    samples = int(total_duration * sample_rate)
    
    t = np.linspace(0, total_duration, samples)
    sound = np.zeros(samples)
    
    # Create adsr lookup
    adsr_lookup = {h[0]: ADSR(h[1], h[2], h[3], h[4]) for h in harmonic_adsrs_tuple}
    default_adsr = ADSR(adsr_attack, adsr_decay, adsr_sustain, adsr_release)
    
    # Add each harmonic
    for i, (h_freq, h_amp) in enumerate(harmonics_tuple):
        this_freq = freq * h_freq
        this_amp = velocity * h_amp
        
        sin_pattern = np.sin(2 * np.pi * this_freq * t)
        harmonic_adsr = adsr_lookup.get(i, default_adsr)
        envelope = harmonic_adsr.get_envelope(duration, sample_rate)
        sound += this_amp * sin_pattern * envelope
    
    return sound

class Instrument:
    def __init__(
        self,
        harmonics: List[Tuple[float, float]],
        adsr: ADSR,
        harmonic_adsrs: Optional[Dict[int, ADSR]] = None
    ):
        self.harmonics = harmonics
        self.adsr = adsr
        self.harmonic_adsrs = harmonic_adsrs or {}
    
    def get_sound(self, freq: float, velocity: float, duration: float, sample_rate: int) -> np.ndarray:
        """Get sound using static cache."""
        # Convert to hashable types for caching
        harmonics_tuple = tuple(self.harmonics)
        harmonic_adsrs_tuple = tuple(
            (i, adsr.attack, adsr.decay, adsr.sustain, adsr.release)
            for i, adsr in self.harmonic_adsrs.items()
        )
        return _cached_get_sound(
            harmonics_tuple,
            self.adsr.attack, self.adsr.decay, self.adsr.sustain, self.adsr.release,
            harmonic_adsrs_tuple,
            freq, velocity, duration, sample_rate
        )

    def mean_amplitudes(self, duration: float, sample_rate: int) -> List[Tuple[float, float]]:
        """Calculate mean amplitudes (cheap computation, no cache needed)."""
    def mean_amplitudes(self, duration: float, sample_rate: int) -> List[Tuple[float, float]]:
        mean_amps = []
        for i, (h_freq, h_amp) in enumerate(self.harmonics):
            harmonic_adsr = self.harmonic_adsrs.get(i) or self.adsr
            mean_amp = harmonic_adsr.mean_amplitude(duration, sample_rate)
            mean_amps.append((h_freq, h_amp * mean_amp))
        return mean_amps

# Use a registry pattern to turn str into default objects
_INSTRUMENT_REGISTRY = {}

def register_instrument(
    name: str,
    factory: Callable[..., Instrument]
):
    _INSTRUMENT_REGISTRY[name] = factory

def get_instrument(
    name_or_instance: Union[str, Instrument],
    **kwargs
) -> Instrument:
    if isinstance(name_or_instance, Instrument):
        return name_or_instance
    if name_or_instance not in _INSTRUMENT_REGISTRY:
        raise ValueError(f"Unknown instrument: {name_or_instance}")
    return _INSTRUMENT_REGISTRY[name_or_instance](**kwargs)

# Register defaults

# Piano has rich harmonic content with inharmonic stretch (upper harmonics are slightly sharp) and a percussive ADSR with quick attack and relatively fast decay.
register_instrument("piano", lambda: Instrument(
    harmonics=[
        (1.000, 1.000),
        (2.002, 0.450),
        (3.005, 0.280),
        (4.010, 0.180),
        (5.015, 0.120),
        (6.025, 0.080),
        (7.035, 0.055),
        (8.050, 0.040)
    ],
    adsr=ADSR(attack=0.005, decay=0.4, sustain=0.3, release=0.5),
    harmonic_adsrs={
        0: ADSR(attack=0.005, decay=0.4, sustain=0.3, release=0.5),
        1: ADSR(attack=0.005, decay=0.35, sustain=0.25, release=0.4),
        2: ADSR(attack=0.005, decay=0.3, sustain=0.2, release=0.35),
        3: ADSR(attack=0.005, decay=0.25, sustain=0.15, release=0.3),
        4: ADSR(attack=0.005, decay=0.2, sustain=0.1, release=0.25),
        5: ADSR(attack=0.005, decay=0.15, sustain=0.08, release=0.2),
        6: ADSR(attack=0.005, decay=0.1, sustain=0.05, release=0.15),
        7: ADSR(attack=0.005, decay=0.08, sustain=0.03, release=0.1)
    }
))

# Guitar has a plucked string sound with characteristic harmonics and a percussive envelope with quick attack and longer sustain than piano.
register_instrument("guitar", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),
        (2.0, 0.55),
        (3.0, 0.35),
        (4.0, 0.22),
        (5.0, 0.15),
        (6.0, 0.10)
    ],
    adsr=ADSR(attack=0.002, decay=0.3, sustain=0.6, release=0.8)
    # No harmonic ADSR overrides
))

# Bass has fewer harmonics than guitar/piano, emphasizing the fundamental and lower harmonics. Long sustain.
register_instrument("bass", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),
        (2.0, 0.40),
        (3.0, 0.20),
        (4.0, 0.10),
        (5.0, 0.05)
    ],
    adsr=ADSR(attack=0.01, decay=0.2, sustain=0.75, release=0.4)
    # No harmonic ADSR overrides
))

# Synth has simple harmonic content (similar to the original default instrument) with no envelope (instant on/off).
register_instrument("synth", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),
        (2.0, 0.50),
        (3.0, 0.33),
        (4.0, 0.25),
        (5.0, 0.20),
        (6.0, 0.17)
    ],
    # No envelope (or instant on / off)
    adsr=ADSR(attack=0.0, decay=0.0, sustain=1.0, release=0.0)
    # No harmonic ADSR overrides
))
