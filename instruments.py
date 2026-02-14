from typing import Optional, Union, Tuple, List, Dict, Callable
from functools import lru_cache
import numpy as np
import torch

# Import TuningSystem for PercussionInstrument
from tuning_systems import TuningSystem, get_tuning_system

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

    def mean_amplitudes(self, freq: float, duration: float, sample_rate: int) -> List[Tuple[float, float]]:
        """Return effective harmonics with amplitudes for dissonance calculation.
        
        Args:
            freq: Fundamental frequency in Hz (used by subclasses, ignored by base Instrument)
            duration: Note duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            List of (frequency_ratio, effective_amplitude) tuples
        """
        mean_amps = []
        for i, (h_freq, h_amp) in enumerate(self.harmonics):
            harmonic_adsr = self.harmonic_adsrs.get(i) or self.adsr
            mean_amp = harmonic_adsr.mean_amplitude(duration, sample_rate)
            mean_amps.append((h_freq, h_amp * mean_amp))
        return mean_amps


class VoiceInstrument(Instrument):
    """Enhanced voice/choral instrument with formant-based synthesis.
    
    Unlike regular instruments where harmonics are integer multiples,
    voices use formants - resonant frequency bands. This creates the
    characteristic vowel sounds (ooh, aah, ee, etc.).
    
    This enhanced version includes:
    1. Pitch-dependent formant shifting (formants move slightly with pitch)
    2. Vibrato (natural pitch modulation)
    3. Breath noise component (aspiration at note attacks)
    4. Inharmonicity (slight detuning of higher harmonics)
    5. More realistic formant shapes
    
    The synthesis works by:
    1. Generate a rich harmonic source with slight inharmonicity
    2. Apply pitch-dependent formant envelope
    3. Add breath noise (filtered noise during attack)
    4. Apply vibrato modulation to the fundamental
    """
    
    def __init__(
        self,
        formants: List[Tuple[float, float, float]],
        adsr: ADSR,
        num_harmonics: int = 24,
        vibrato_rate: float = 5.5,      # Hz, typical vibrato rate
        vibrato_depth: float = 0.03,     # +/- 3% pitch variation
        breath_amount: float = 0.15,     # Amount of breath noise
        inharmonicity: float = 0.001,    # Higher harmonics slightly sharp
        formant_shift_rate: float = 0.15  # Formants shift 15% per octave
    ):
        """Initialize voice instrument with enhanced synthesis.
        
        Args:
            formants: List of (frequency_hz, amplitude, bandwidth_hz) tuples
                     These are BASE frequencies; they shift with pitch
            adsr: ADSR envelope for the voice
            num_harmonics: Number of harmonics to generate (default 24)
            vibrato_rate: Vibrato frequency in Hz (default 5.5)
            vibrato_depth: Pitch modulation depth as ratio (default 0.03 = +/-3%)
            breath_amount: Amount of breath noise 0-1 (default 0.15)
            inharmonicity: Coefficient for harmonic stretch (default 0.001)
            formant_shift_rate: How much formants shift per octave (default 0.15)
        """
        # Store formants and synthesis parameters
        self.base_formants = formants  # Base formant frequencies
        self.num_harmonics = num_harmonics
        self.vibrato_rate = vibrato_rate
        self.vibrato_depth = vibrato_depth
        self.breath_amount = breath_amount
        self.inharmonicity = inharmonicity
        self.formant_shift_rate = formant_shift_rate
        
        # Reference pitch for formant scaling (A3 = 220 Hz)
        self.reference_freq = 220.0
        
        # Base class needs harmonics - we'll override mean_amplitudes
        dummy_harmonics = [(float(i), 1.0 / i) for i in range(1, num_harmonics + 1)]
        
        super().__init__(
            harmonics=dummy_harmonics,
            adsr=adsr,
            harmonic_adsrs={}
        )
    
    def _get_shifted_formants(self, freq: float) -> List[Tuple[float, float, float]]:
        """Get formant frequencies shifted based on pitch.
        
        Real voices shift formants as pitch changes (the "whoop" effect).
        Higher pitches = higher formants, but less than proportionally.
        
        Args:
            freq: Fundamental frequency in Hz
            
        Returns:
            List of (shifted_freq, amplitude, bandwidth) tuples
        """
        # Calculate how many octaves above reference
        octaves = np.log2(freq / self.reference_freq)
        
        shifted_formants = []
        for formant_freq, formant_amp, bandwidth in self.base_formants:
            # Formants shift with pitch but less than proportionally
            # The shift_rate controls how much (0.15 = 15% per octave)
            shift_factor = 1.0 + self.formant_shift_rate * octaves
            shifted_freq = formant_freq * shift_factor
            
            # Bandwidth also increases slightly with frequency
            shifted_bandwidth = bandwidth * (1.0 + 0.1 * octaves)
            
            shifted_formants.append((shifted_freq, formant_amp, shifted_bandwidth))
        
        return shifted_formants
    
    def _compute_harmonic_frequency(self, n: int, fundamental: float) -> float:
        """Compute the frequency of the nth harmonic with inharmonicity.
        
        Real voices have slightly stretched harmonics due to vocal fold
        tension. Higher harmonics are slightly sharp.
        
        Args:
            n: Harmonic number (1 = fundamental)
            fundamental: Fundamental frequency
            
        Returns:
            Frequency of the nth harmonic with inharmonicity applied
        """
        # Inharmonicity formula: f_n = n * f0 * sqrt(1 + B * n^2)
        # where B is the inharmonicity coefficient
        stretch = np.sqrt(1.0 + self.inharmonicity * n * n)
        return fundamental * n * stretch
    
    def _compute_formant_envelope(self, freq: float) -> List[Tuple[float, float]]:
        """Compute effective harmonics with pitch-dependent formant weighting.
        
        Args:
            freq: Fundamental frequency in Hz
            
        Returns:
            List of (frequency_ratio, effective_amplitude) tuples
        """
        effective_harmonics = []
        shifted_formants = self._get_shifted_formants(freq)
        
        for n in range(1, self.num_harmonics + 1):
            harmonic_freq = self._compute_harmonic_frequency(n, freq)
            
            # Source spectrum: 1/n falloff but with slight rolloff at very high freqs
            base_amp = 1.0 / n
            
            # Additional rolloff above ~4kHz (vocal tract can't support high freqs well)
            if harmonic_freq > 4000:
                base_amp *= np.exp(-(harmonic_freq - 4000) / 2000)
            
            # Apply formant envelope using more realistic shape
            formant_weight = 0.0
            for formant_freq, formant_amp, bandwidth in shifted_formants:
                # Use a combination of Gaussian and Lorentzian for more realistic shape
                distance = abs(harmonic_freq - formant_freq)
                
                # Gaussian component (smooth peak)
                gauss_weight = np.exp(-0.5 * (distance / bandwidth) ** 2)
                
                # Lorentzian component (broader tails like real resonances)
                lorentz_weight = (bandwidth ** 2) / (distance ** 2 + bandwidth ** 2)
                
                # Combine them (70% Gaussian, 30% Lorentzian for natural sound)
                combined_weight = 0.7 * gauss_weight + 0.3 * lorentz_weight
                formant_weight += formant_amp * combined_weight
            
            # Ensure some minimum amplitude so voice isn't silent
            formant_weight = max(formant_weight, 0.05)
            effective_amp = base_amp * formant_weight
            
            effective_harmonics.append((float(n), effective_amp))
        
        return effective_harmonics
    
    def get_sound(self, freq: float, velocity: float, duration: float, sample_rate: int) -> np.ndarray:
        """Generate enhanced voice sound with vibrato, noise, and formants."""
        max_release = self.adsr.release
        total_duration = duration + max_release
        samples = int(duration * sample_rate) + int(max_release * sample_rate)
        
        t = np.linspace(0, total_duration, samples)
        sound = np.zeros(samples)
        
        envelope = self.adsr.get_envelope(duration, sample_rate)
        envelope_end = envelope.shape[0]
        
        # Generate vibrato modulation
        vibrato_phase = 2 * np.pi * self.vibrato_rate * t
        vibrato_factor = 1.0 + self.vibrato_depth * np.sin(vibrato_phase)
        
        # Generate breath noise (filtered white noise)
        # More breath during attack, less during sustain
        noise = np.random.randn(samples) * self.breath_amount * velocity
        
        # Breath envelope: strongest at attack, decays quickly
        breath_attack = int(0.08 * sample_rate)  # 80ms breath burst
        breath_envelope = np.zeros(samples)
        if breath_attack > 0:
            breath_envelope[:min(breath_attack, samples)] = np.linspace(1.0, 0.0, min(breath_attack, samples))
        # Also modulate by main envelope but with faster decay
        breath_envelope *= np.exp(-t / 0.15)  # 150ms decay
        
        # Filter breath noise to be high-frequency (simulating aspiration)
        # Simple high-pass characteristic: emphasize frequencies above 1kHz
        breath_filtered = noise * breath_envelope
        
        # Get effective harmonics with formant weighting
        effective_harmonics = self._compute_formant_envelope(freq)
        
        # Add each formant-weighted harmonic with vibrato
        for n, (h_ratio, h_amp) in enumerate(effective_harmonics, 1):
            # Compute actual harmonic frequency with inharmonicity
            base_harmonic_freq = self._compute_harmonic_frequency(n, freq)
            
            # Apply vibrato to each harmonic (more vibrato on higher harmonics)
            # Real voices have more vibrato excursion on higher harmonics
            harmonic_vibrato = 1.0 + (self.vibrato_depth * (1.0 + 0.1 * n)) * np.sin(vibrato_phase)
            modulated_freq = base_harmonic_freq * harmonic_vibrato
            
            # Integrate frequency for phase (FM synthesis)
            phase = 2 * np.pi * np.cumsum(modulated_freq / sample_rate) * (1.0 / sample_rate) * sample_rate
            phase = phase % (2 * np.pi)
            
            sin_pattern = np.sin(phase)
            this_amp = velocity * h_amp
            
            sound[:envelope_end] += this_amp * sin_pattern[:envelope_end] * envelope
        
        # Add breath noise (mostly during attack)
        sound[:envelope_end] += breath_filtered[:envelope_end] * envelope
        
        return sound
    
    def mean_amplitudes(self, freq: float, duration: float, sample_rate: int) -> List[Tuple[float, float]]:
        """Return formant-weighted harmonics for dissonance calculation.
        
        For loss calculation, we use the average (non-vibrato) state.
        """
        effective_harmonics = self._compute_formant_envelope(freq)
        mean_amp_factor = self.adsr.mean_amplitude(duration, sample_rate)
        
        # Apply ADSR mean amplitude to all harmonics
        return [(ratio, amp * mean_amp_factor) for ratio, amp in effective_harmonics]


class PercussionInstrument(Instrument):
    """MIDI Channel 10-style percussion instrument.
    
    Unlike pitched instruments where notes are frequencies, percussion
    uses note numbers (0-127 on Channel 10) to represent different drum
    types. Each drum type has its own fixed-frequency partials that
    don't track any pitch parameter.
    
    The incoming frequency is converted to a MIDI note number via a
    tuning system, then looked up in the drum profile dictionary.
    """
    
    def __init__(
        self,
        drum_profiles: Dict[int, List[Tuple[float, float]]],
        adsr_profiles: Dict[int, ADSR],
        tuning_system  # TuningSystem for freq <-> note conversion
    ):
        """Initialize percussion instrument with drum profiles.
        
        Args:
            drum_profiles: Dict mapping MIDI note numbers to lists of
                          (frequency_hz, amplitude) tuples. These are
                          ABSOLUTE frequencies, not ratios.
            adsr_profiles: Dict mapping MIDI note numbers to ADSR envelopes
            tuning_system: TuningSystem for converting frequencies to MIDI notes
        """
        self.drum_profiles = drum_profiles
        self._adsr_profiles = adsr_profiles
        self._tuning_system = tuning_system
        
        # Create dummy harmonics for base class compatibility
        # Real harmonics come from drum_profiles lookup
        dummy_harmonics = [(1.0, 1.0)]

        # Create dummy ADSR for base class compatibility
        dummy_adsr = ADSR(attack=0.001, decay=0.1, sustain=0.0, release=0.05)
        
        super().__init__(
            harmonics=dummy_harmonics,
            adsr=dummy_adsr,
            harmonic_adsrs={}
        )
    
    def _freq_to_note(self, freq: float) -> int:
        """Convert frequency to MIDI note number using tuning system."""
        # Use the tuning system's freq_to_key method
        note = self._tuning_system.freq_to_key(freq)
        if note is None:
            # Fallback: compute from 12-TET formula
            note = int(69 + 12 * np.log2(freq / 440.0))
        return note
    
    def _get_drum_partials(self, freq: float) -> List[Tuple[float, float]]:
        """Get the partials for a drum given the frequency parameter.
        
        Args:
            freq: The frequency parameter (typically from tuning_system.key_to_freq(note))
            
        Returns:
            List of (frequency_ratio, amplitude) tuples where ratios are
            relative to the incoming freq parameter for dissonance calculation
        """
        note = self._freq_to_note(freq)
        
        if note not in self.drum_profiles:
            # Return silence for undefined drum notes
            return [(1.0, 0.0)]
        
        partials = self.drum_profiles[note]
        
        # Convert absolute frequencies to ratios relative to freq parameter
        # This allows the dissonance matrix to compare drums to pitched instruments
        # ratio = partial_freq / freq, so if freq=61.375 and partial=60, ratio≈0.977
        ratio_partials = []
        for partial_freq, amplitude in partials:
            ratio = partial_freq / freq if freq > 0 else 1.0
            ratio_partials.append((ratio, amplitude))
        
        return ratio_partials
    
    def _get_drum_adsr(self, freq: float) -> ADSR:
        """Get the ADSR envelope for a drum given the frequency parameter."""
        note = self._freq_to_note(freq)
        return self._adsr_profiles.get(note, self.adsr) # Use the dummy ADSR if no specific profile is defined
    
    def get_sound(self, freq: float, velocity: float, duration: float, sample_rate: int) -> np.ndarray:
        """Generate drum sound with fixed-frequency partials."""
        partials = self._get_drum_partials(freq)
        adsr = self._get_drum_adsr(freq)
        
        max_release = adsr.release
        total_duration = duration + max_release
        samples = int(duration * sample_rate) + int(max_release * sample_rate)
        
        t = np.linspace(0, total_duration, samples)
        sound = np.zeros(samples)
        
        envelope = adsr.get_envelope(duration, sample_rate)
        envelope_end = envelope.shape[0]
        
        # Add each partial at its fixed frequency
        for ratio, amp in partials:
            # For sound generation, convert ratio back to absolute frequency
            # ratio = partial_freq / freq, so partial_freq = ratio * freq
            partial_freq = ratio * freq
            this_amp = velocity * amp
            
            sin_pattern = np.sin(2 * np.pi * partial_freq * t)
            sound[:envelope_end] += this_amp * sin_pattern[:envelope_end] * envelope
        
        return sound
    
    def mean_amplitudes(self, freq: float, duration: float, sample_rate: int) -> List[Tuple[float, float]]:
        """Return drum partials as ratios for dissonance calculation."""
        partials = self._get_drum_partials(freq)
        adsr = self._get_drum_adsr(freq)
        mean_amp_factor = adsr.mean_amplitude(duration, sample_rate)
        
        return [(ratio, amp * mean_amp_factor) for ratio, amp in partials]

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

# 🎸🎹🥁🍜 4=1

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

# Standard drum kit with Acoustic Bass Drum (35), Acoustic Snare (38), and Closed Hi Hat (42)
# Uses 12-TET tuning system to map frequencies to MIDI note numbers
register_instrument("percussion", lambda: PercussionInstrument(
    drum_profiles={
        35: [  # Acoustic Bass Drum (B1 = 61.375 Hz in 12-TET, but these are arbitrary)
            (60.0, 1.00), # Fundamental "boom"
            (90.0, 0.30), # Inharmonic click
            (132.0, 0.20), # Body resonance
            (210.0, 0.15), # Click/harmonics
        ],
        38: [  # Acoustic Snare (D2) These are approximate - snare is more noise than tones
            (200.0, 0.80), # Fundamental (drum body)
            (282.0, 0.60),
            (346.0, 0.50),
            (446.0, 0.40),
            (566.0, 0.35),
            (692.0, 0.30),
        ],
        42: [ # Closed Hi Hat (F#2) Modeled as many closely-spaced high harmonics
            (185.0, 0.50), # Higher relative to fundamental
            (277.5, 0.60),
            (370.0, 0.55),
            (462.5, 0.50),
            (555.0, 0.45),
            (647.5, 0.40),
            (740.0, 0.35),
        ],
    },
    adsr_profiles={
        35: ADSR(attack=0.001, decay=0.15, sustain=0.0, release=0.05),
        38: ADSR(attack=0.001, decay=0.12, sustain=0.0, release=0.08),
        42: ADSR(attack=0.001, decay=0.05, sustain=0.0, release=0.02),
    },
    tuning_system=get_tuning_system("12-TET")
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

# Keyboards

# Rhodes/Electric Piano - tine-based with bell-like attack
register_instrument("rhodes", lambda: Instrument(
    harmonics=[
        (1.00, 1.00),
        (2.00, 0.40),
        (3.00, 0.15),
        (4.50, 0.10), # Inharmonic overtone (characteristic of tines)
        (5.50, 0.08), # Inharmonic overtone
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
        (1.00, 1.00), # 8' stop (fundamental)
        (2.00, 0.70), # 4' stop (octave)
        (3.00, 0.50), # 2 2/3' stop (twelfth)
        (4.00, 0.40), # 2' stop (fifteenth)
        (5.00, 0.25), # 1 3/5' stop (seventeenth)
        (6.00, 0.15), # 1 1/3' stop (nineteenth)
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

# Bowed Strings

# Violin - bowed string with sustained envelope and rich harmonics
register_instrument("violin", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),
        (2.0, 0.60),
        (3.0, 0.40),
        (4.0, 0.28),
        (5.0, 0.18),
        (6.0, 0.12),
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
        (1.0, 1.00),
        (2.0, 0.55),
        (3.0, 0.30),
        (4.0, 0.15),
        (5.0, 0.08),
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

# Brass

# Trumpet - bright brass with strong upper harmonics
register_instrument("trumpet", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),
        (2.0, 0.80), # 2nd harmonic (strong - characteristic of brass)
        (3.0, 0.65),
        (4.0, 0.50),
        (5.0, 0.35),
        (6.0, 0.20),
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

# Woodwinds

# Flute - breathy, mostly fundamental with odd harmonics
register_instrument("flute", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),
        (2.0, 0.25), # 2nd harmonic (weak in flute)
        (3.0, 0.35),
        (4.0, 0.10),
        (5.0, 0.15),
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
        (1.0, 1.00),
        (3.0, 0.55), # 3rd harmonic (strong - cylindrical bore)
        (5.0, 0.30),
        (7.0, 0.15),
        (9.0, 0.08),
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
        (1.0, 1.00),
        (2.0, 0.70),
        (3.0, 0.50),
        (4.0, 0.35),
        (5.0, 0.22),
        (6.0, 0.12),
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

# Synth / Electronic

# Pure sine tone
# Used by the midi "instrument"
register_instrument("sine", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),
    ],
    adsr=ADSR(attack=0.0, decay=0.0, sustain=1.0, release=0.0)
    # No harmonic ADSR overrides
))

# The default instrument from v1. Apparently it was a sawtooth wave the whole time
# Not used by any instrument now
register_instrument("default", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),
        (2.0, 0.50),
        (3.0, 0.33),
        (4.0, 0.25),
        (5.0, 0.20),
        (6.0, 0.17),
    ],
    adsr=ADSR(attack=0.0, decay=0.0, sustain=1.0, release=0.0) # No envelope
    # No harmonic ADSR overrides
))

# Sawtooth wave with all harmonics (1/n amplitude)
register_instrument("saw", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),
        (2.0, 0.50),
        (3.0, 0.333),
        (4.0, 0.25),
        (5.0, 0.20),
        (6.0, 0.167),
        (7.0, 0.143),
        (8.0, 0.125),
    ],
    adsr=ADSR(attack=0.01, decay=0.2, sustain=0.7, release=0.4),
    # Classic synth envelope
))

# Square wave with odd harmonics only (1/n amplitude)
register_instrument("square", lambda: Instrument(
    harmonics=[
        (1.0, 1.00),
        (3.0, 0.333),
        (5.0, 0.20),
        (7.0, 0.143),
        (9.0, 0.111),
    ],
    adsr=ADSR(attack=0.01, decay=0.2, sustain=0.7, release=0.4),
    # Classic chiptune/synth envelope
))

# Idiophones

# Music Box - clear bell-like tones with distinct harmonics
register_instrument("music_box", lambda: Instrument(
    harmonics=[
        (1.00, 1.00),
        (2.76, 0.35), # Inharmonic overtone (characteristic of music boxes)
        (5.40, 0.18), # Another inharmonic overtone
        (8.93, 0.08), # Higher overtone
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
        (1.00, 1.00),
        (4.00, 0.45), # 4th harmonic (strong in vibraphone)
        (6.25, 0.15), # Inharmonic overtone
        (10.0, 0.25), # 10th harmonic
    ],
    adsr=ADSR(attack=0.005, decay=0.4, sustain=0.6, release=1.0),
    # Sharp attack, long sustain with motor-driven tremolo
    harmonic_adsrs={
        0: ADSR(attack=0.005, decay=0.4, sustain=0.6, release=1.0),
        1: ADSR(attack=0.003, decay=0.3, sustain=0.5, release=0.8),
        2: ADSR(attack=0.002, decay=0.15, sustain=0.3, release=0.4),
        3: ADSR(attack=0.002, decay=0.2, sustain=0.4, release=0.6),
    }
))

# Choir / Voice

# Enhanced formant values based on vocal research
# Format: (frequency_hz, amplitude, bandwidth_hz)
# F1, F2, F3, F4 are the main formants that define vowel quality

# Choir "ooh" - rounded vowel with low first formant
# Typical formants for /u/ (oo) vowel: F1~300, F2~700, F3~2300, F4~3200
register_instrument("voice_ooh", lambda: VoiceInstrument(
    formants=[
        (300, 1.00, 60),    # F1: low frequency, creates "ooh" darkness
        (700, 0.55, 90),    # F2: mid frequency, close to F1
        (2300, 0.30, 140),  # F3: high frequency
        (3200, 0.15, 200),  # F4: very high, adds "brightness"
    ],
    adsr=ADSR(attack=0.18, decay=0.12, sustain=0.82, release=0.55),
    num_harmonics=24,
    vibrato_rate=5.2,
    vibrato_depth=0.025,
    breath_amount=0.12,
    inharmonicity=0.0008,
    formant_shift_rate=0.12
))

# Choir "aah" - open vowel with higher first formant
# Typical formants for /a/ (ah) vowel: F1~750, F2~1200, F3~2600, F4~3500
register_instrument("voice_aah", lambda: VoiceInstrument(
    formants=[
        (750, 1.00, 85),    # F1: higher frequency, creates "aah" brightness
        (1150, 0.70, 110),  # F2: mid-high, well separated from F1
        (2650, 0.35, 160),  # F3: high frequency
        (3500, 0.18, 220),  # F4: very high
    ],
    adsr=ADSR(attack=0.15, decay=0.10, sustain=0.85, release=0.50),
    num_harmonics=24,
    vibrato_rate=5.5,
    vibrato_depth=0.030,
    breath_amount=0.15,
    inharmonicity=0.0010,
    formant_shift_rate=0.15
))

# Choir "eeh" - bright vowel with very high F2
# Typical formants for /i/ (ee) vowel: F1~280, F2~2200, F3~3000, F4~3500
register_instrument("voice_eeh", lambda: VoiceInstrument(
    formants=[
        (280, 1.00, 55),    # F1: very low
        (2200, 0.80, 130),  # F2: very high, creates "eeh" brightness
        (3000, 0.40, 170),  # F3: high
        (3600, 0.20, 230),  # F4: very high
    ],
    adsr=ADSR(attack=0.12, decay=0.08, sustain=0.87, release=0.45),
    num_harmonics=24,
    vibrato_rate=5.8,
    vibrato_depth=0.035,
    breath_amount=0.10,
    inharmonicity=0.0012,
    formant_shift_rate=0.18
))
