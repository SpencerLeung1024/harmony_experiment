from typing import Optional, List, Dict
from functools import cache
import numpy as np
import torch


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
    
    @cache
    def get_envelope(self, duration: float, sample_rate: int) -> np.ndarray:
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
    
    @cache
    def mean_amplitude(self, duration: float, sample_rate: int) -> float:
        envelope = self.get_envelope(duration, sample_rate)
        return np.mean(envelope)

class Instrument:
    def __init__(
        self,
        harmonics: List[(float, float)],
        adsr: ADSR,
        harmonic_adsrs: Optional[Dict[int, ADSR]] = {}
    ):
        self.harmonics = harmonics
        self.adsr = adsr
        self.harmonic_adsrs = harmonic_adsrs
    
    @cache
    def get_sound(self, freq: float, velocity: float, duration: float, sample_rate: int) -> np.ndarray:
        # Find out which harmonic has the longest release time
        max_release = max([self.adsr.release] + [adsr.release for adsr in self.harmonic_adsrs.values()])

        total_duration = duration + max_release
        samples = int(total_duration * sample_rate)
        
        t = np.linspace(0, total_duration, samples)
        sound = np.zeros(samples)

        # Add each harmonic
        for i, (h_freq, h_amp) in enumerate(self.harmonics):
            this_freq = freq * h_freq
            this_amp = velocity * h_amp

            sin_pattern = np.sin(2 * np.pi * this_freq * t)
            harmonic_adsr = self.harmonic_adsrs.get(i) or self.adsr
            envelope = harmonic_adsr.get_envelope(duration, sample_rate)
            sound += this_amp * sin_pattern * envelope
        
        return sound

    @cache
    def mean_amplitudes(self, duration: float, sample_rate: int) -> List[(float, float)]:
        mean_amps = []
        for i, (h_freq, h_amp) in enumerate(self.harmonics):
            harmonic_adsr = self.harmonic_adsrs.get(i) or self.adsr
            mean_amp = harmonic_adsr.mean_amplitude(duration, sample_rate)
            mean_amps.append((h_freq, h_amp * mean_amp))
        return mean_amps
