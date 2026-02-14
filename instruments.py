from typing import Optional, Union, Tuple, List, Dict, Callable
from functools import lru_cache
import numpy as np
import torch

# Module-level cached functions for static caching
# Kimi K2.5 said putting @cache on object methods will lead to memory leaks since self is always different
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
    #samples = int(total_duration * sample_rate)
    # Pernicious off by one error
    samples = int(duration * sample_rate) + int(max_release * sample_rate)
    
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
        envelope_end = envelope.shape[0]
        # Harmonics may have shorter release times so trim the sin_pattern to the envelope length
        sound[:envelope_end] += this_amp * sin_pattern[:envelope_end] * envelope
    
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

# The default instrument from v1. I guess you could call it a synth but it's more of a beeper. Has no envelope (instant on/off).
register_instrument("default", lambda: Instrument(
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

# Pure sine tone.
register_instrument("sine", lambda: Instrument(
    harmonics=[
        (1.0, 1.00)
    ],
    adsr=ADSR(attack=0.0, decay=0.0, sustain=1.0, release=0.0)
    # No harmonic ADSR overrides
))

# ==================== BOWED STRINGS ====================

# Violin - bowed string with sustained envelope and rich harmonics
register_instrument("violin", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),   # Fundamental (strong)
        (2.0, 0.60),   # 2nd harmonic
        (3.0, 0.40),   # 3rd harmonic
        (4.0, 0.28),   # 4th harmonic
        (5.0, 0.18),   # 5th harmonic
        (6.0, 0.12),   # 6th harmonic
    ],
    adsr=ADSR(attack=0.15, decay=0.1, sustain=0.85, release=0.4),
    # Bowed strings have gradual attack (bow engagement), long sustain
    harmonic_adsrs={
        0: ADSR(attack=0.15, decay=0.1, sustain=0.85, release=0.4),
        1: ADSR(attack=0.12, decay=0.1, sustain=0.75, release=0.35),
        2: ADSR(attack=0.10, decay=0.08, sustain=0.65, release=0.3),
        3: ADSR(attack=0.08, decay=0.06, sustain=0.55, release=0.25),
        4: ADSR(attack=0.06, decay=0.05, sustain=0.45, release=0.2),
        5: ADSR(attack=0.05, decay=0.04, sustain=0.35, release=0.15),
    }
))

# Cello - warm bowed string with emphasis on fundamental
register_instrument("cello", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),   # Fundamental (very strong - warm sound)
        (2.0, 0.55),   # 2nd harmonic
        (3.0, 0.30),   # 3rd harmonic
        (4.0, 0.15),   # 4th harmonic
        (5.0, 0.08),   # 5th harmonic
    ],
    adsr=ADSR(attack=0.18, decay=0.12, sustain=0.88, release=0.5),
    # Slower attack than violin, very long sustain
    harmonic_adsrs={
        0: ADSR(attack=0.18, decay=0.12, sustain=0.88, release=0.5),
        1: ADSR(attack=0.15, decay=0.1, sustain=0.78, release=0.45),
        2: ADSR(attack=0.12, decay=0.08, sustain=0.68, release=0.4),
        3: ADSR(attack=0.10, decay=0.06, sustain=0.58, release=0.35),
        4: ADSR(attack=0.08, decay=0.05, sustain=0.48, release=0.3),
    }
))

# ==================== KEYBOARDS ====================

# Rhodes/Electric Piano - tine-based with bell-like attack
register_instrument("rhodes", lambda: Instrument(
    harmonics=[
        (1.00, 1.00),   # Fundamental
        (2.00, 0.40),   # 2nd harmonic
        (3.00, 0.15),   # 3rd harmonic
        (4.50, 0.10),   # Inharmonic overtone (characteristic of tines)
        (5.50, 0.08),   # Inharmonic overtone
    ],
    adsr=ADSR(attack=0.005, decay=0.6, sustain=0.35, release=0.8),
    # Quick attack (tine strike), long decay with sustain
    harmonic_adsrs={
        0: ADSR(attack=0.005, decay=0.6, sustain=0.35, release=0.8),
        1: ADSR(attack=0.003, decay=0.45, sustain=0.25, release=0.6),
        2: ADSR(attack=0.002, decay=0.3, sustain=0.15, release=0.4),
        3: ADSR(attack=0.001, decay=0.2, sustain=0.10, release=0.3),
        4: ADSR(attack=0.001, decay=0.15, sustain=0.05, release=0.2),
    }
))

# Organ - pipe organ with rich harmonic ranks (8', 4', 2 2/3', 2', etc.)
register_instrument("organ", lambda: Instrument(
    harmonics=[
        (1.00, 1.00),   # 8' stop (fundamental)
        (2.00, 0.70),   # 4' stop (octave)
        (3.00, 0.50),   # 2 2/3' stop (twelfth)
        (4.00, 0.40),   # 2' stop (fifteenth)
        (5.00, 0.25),   # 1 3/5' stop (seventeenth)
        (6.00, 0.15),   # 1 1/3' stop (nineteenth)
    ],
    adsr=ADSR(attack=0.03, decay=0.0, sustain=1.0, release=0.08),
    # Instant full sustain (air flow), quick release when key released
    harmonic_adsrs={
        0: ADSR(attack=0.03, decay=0.0, sustain=1.0, release=0.08),
        1: ADSR(attack=0.025, decay=0.0, sustain=0.95, release=0.07),
        2: ADSR(attack=0.02, decay=0.0, sustain=0.90, release=0.06),
        3: ADSR(attack=0.02, decay=0.0, sustain=0.85, release=0.06),
        4: ADSR(attack=0.015, decay=0.0, sustain=0.80, release=0.05),
        5: ADSR(attack=0.015, decay=0.0, sustain=0.75, release=0.05),
    }
))

# ==================== BRASS ====================

# Trumpet - bright brass with strong upper harmonics
register_instrument("trumpet", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),   # Fundamental
        (2.0, 0.80),   # 2nd harmonic (strong - characteristic of brass)
        (3.0, 0.65),   # 3rd harmonic
        (4.0, 0.50),   # 4th harmonic
        (5.0, 0.35),   # 5th harmonic
        (6.0, 0.20),   # 6th harmonic
    ],
    adsr=ADSR(attack=0.08, decay=0.15, sustain=0.75, release=0.3),
    # Moderate attack (lip buzzing starts), good sustain
    harmonic_adsrs={
        0: ADSR(attack=0.08, decay=0.15, sustain=0.75, release=0.3),
        1: ADSR(attack=0.06, decay=0.12, sustain=0.70, release=0.28),
        2: ADSR(attack=0.05, decay=0.10, sustain=0.65, release=0.25),
        3: ADSR(attack=0.04, decay=0.08, sustain=0.60, release=0.22),
        4: ADSR(attack=0.03, decay=0.06, sustain=0.55, release=0.20),
        5: ADSR(attack=0.03, decay=0.05, sustain=0.50, release=0.18),
    }
))

# ==================== WOODWINDS ====================

# Flute - breathy, mostly fundamental with odd harmonics
register_instrument("flute", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),   # Fundamental (very strong)
        (2.0, 0.25),   # 2nd harmonic (weak in flute)
        (3.0, 0.35),   # 3rd harmonic
        (4.0, 0.10),   # 4th harmonic
        (5.0, 0.15),   # 5th harmonic
    ],
    adsr=ADSR(attack=0.06, decay=0.1, sustain=0.80, release=0.25),
    # Moderate attack (air flow), breathy sustain
    harmonic_adsrs={
        0: ADSR(attack=0.06, decay=0.1, sustain=0.80, release=0.25),
        1: ADSR(attack=0.05, decay=0.08, sustain=0.60, release=0.20),
        2: ADSR(attack=0.04, decay=0.06, sustain=0.55, release=0.18),
        3: ADSR(attack=0.03, decay=0.05, sustain=0.50, release=0.15),
        4: ADSR(attack=0.03, decay=0.04, sustain=0.45, release=0.12),
    }
))

# Clarinet - hollow sound, strong odd harmonics only
register_instrument("clarinet", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),   # Fundamental (strong)
        (3.0, 0.55),   # 3rd harmonic (strong - cylindrical bore)
        (5.0, 0.30),   # 5th harmonic
        (7.0, 0.15),   # 7th harmonic
        (9.0, 0.08),   # 9th harmonic
    ],
    adsr=ADSR(attack=0.05, decay=0.12, sustain=0.78, release=0.2),
    # Quick attack (reed), good sustain
    harmonic_adsrs={
        0: ADSR(attack=0.05, decay=0.12, sustain=0.78, release=0.2),
        1: ADSR(attack=0.04, decay=0.10, sustain=0.68, release=0.18),
        2: ADSR(attack=0.03, decay=0.08, sustain=0.58, release=0.15),
        3: ADSR(attack=0.03, decay=0.06, sustain=0.48, release=0.12),
        4: ADSR(attack=0.02, decay=0.05, sustain=0.38, release=0.10),
    }
))

# Saxophone - bright reed instrument with full spectrum
register_instrument("saxophone", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),   # Fundamental
        (2.0, 0.70),   # 2nd harmonic
        (3.0, 0.50),   # 3rd harmonic
        (4.0, 0.35),   # 4th harmonic
        (5.0, 0.22),   # 5th harmonic
        (6.0, 0.12),   # 6th harmonic
    ],
    adsr=ADSR(attack=0.04, decay=0.1, sustain=0.82, release=0.25),
    # Fast attack (reed), bright sustain
    harmonic_adsrs={
        0: ADSR(attack=0.04, decay=0.1, sustain=0.82, release=0.25),
        1: ADSR(attack=0.03, decay=0.08, sustain=0.75, release=0.22),
        2: ADSR(attack=0.03, decay=0.07, sustain=0.68, release=0.20),
        3: ADSR(attack=0.025, decay=0.06, sustain=0.60, release=0.18),
        4: ADSR(attack=0.02, decay=0.05, sustain=0.52, release=0.15),
        5: ADSR(attack=0.02, decay=0.04, sustain=0.45, release=0.12),
    }
))

# ==================== SYNTH/ELECTRONIC ====================

# Saw Lead - sawtooth wave with all harmonics (1/n amplitude)
register_instrument("saw_lead", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),    # 1/1
        (2.0, 0.50),    # 1/2
        (3.0, 0.333),   # 1/3
        (4.0, 0.25),    # 1/4
        (5.0, 0.20),    # 1/5
        (6.0, 0.167),   # 1/6
        (7.0, 0.143),   # 1/7
        (8.0, 0.125),   # 1/8
    ],
    adsr=ADSR(attack=0.01, decay=0.2, sustain=0.7, release=0.4),
    # Classic synth envelope
))

# Square Lead - square wave with odd harmonics only (1/n amplitude)
register_instrument("square_lead", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),    # 1/1
        (3.0, 0.333),   # 1/3
        (5.0, 0.20),    # 1/5
        (7.0, 0.143),   # 1/7
        (9.0, 0.111),   # 1/9
    ],
    adsr=ADSR(attack=0.01, decay=0.2, sustain=0.7, release=0.4),
    # Classic chiptune/synth envelope
))

# ==================== IDIOPHONES/PERCUSSION ====================

# Music Box - clear bell-like tones with distinct harmonics
register_instrument("music_box", lambda: Instrument(
    harmonics=[
        (1.00, 1.00),   # Fundamental
        (2.76, 0.35),   # Inharmonic overtone (characteristic of music boxes)
        (5.40, 0.18),   # Another inharmonic overtone
        (8.93, 0.08),   # Higher overtone
    ],
    adsr=ADSR(attack=0.001, decay=0.8, sustain=0.0, release=0.6),
    # Instant attack, long decay, no sustain (plucked metal)
    harmonic_adsrs={
        0: ADSR(attack=0.001, decay=0.8, sustain=0.0, release=0.6),
        1: ADSR(attack=0.001, decay=0.5, sustain=0.0, release=0.4),
        2: ADSR(attack=0.001, decay=0.3, sustain=0.0, release=0.25),
        3: ADSR(attack=0.001, decay=0.15, sustain=0.0, release=0.15),
    }
))

# Vibraphone - struck metal bars with tremolo
register_instrument("vibraphone", lambda: Instrument(
    harmonics=[
        (1.00, 1.00),   # Fundamental
        (4.00, 0.45),   # 4th harmonic (strong in vibraphone)
        (10.0, 0.25),   # 10th harmonic
        (6.25, 0.15),   # Inharmonic overtone
    ],
    adsr=ADSR(attack=0.005, decay=0.4, sustain=0.6, release=1.0),
    # Sharp attack, long sustain with motor-driven tremolo
    harmonic_adsrs={
        0: ADSR(attack=0.005, decay=0.4, sustain=0.6, release=1.0),
        1: ADSR(attack=0.003, decay=0.3, sustain=0.5, release=0.8),
        2: ADSR(attack=0.002, decay=0.2, sustain=0.4, release=0.6),
        3: ADSR(attack=0.002, decay=0.15, sustain=0.3, release=0.4),
    }
))
