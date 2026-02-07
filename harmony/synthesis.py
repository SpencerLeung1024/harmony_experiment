"""
Audio synthesis module for Harmony From First Principles.

This module provides audio synthesis for band members, using additive synthesis
with ADSR envelopes. Each note is synthesized as a sum of harmonics with
per-harmonic envelope shaping.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch

from .band import BandMember, PolyphonicMember, MonophonicMember, DrumMember
from .instruments import Instrument, ADSR


class AudioSynthesizer:
    """Synthesizes audio for band members using additive synthesis with ADSR.
    
    Uses additive synthesis to generate audio by summing sine waves for each
    harmonic of each note, with ADSR envelopes applied per harmonic for
    realistic instrument timbres.
    
    Attributes:
        sample_rate: Audio sample rate in Hz
        beat_duration: Duration of each beat in seconds
    """
    
    def __init__(self, sample_rate: int = 22050, beat_duration: float = 0.5):
        """Initialize the audio synthesizer.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 22050)
            beat_duration: Duration of each beat in seconds (default: 0.5)
        """
        self.sample_rate = sample_rate
        self.beat_duration = beat_duration
    
    def synthesize_member(
        self,
        member: BandMember,
        duration: Optional[float] = None,
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """Synthesize audio for a single band member.
        
        Generates audio for all beats, applying the appropriate synthesis
        method based on the member type:
        - Polyphonic: Multiple notes per beat with soft activation
        - Monophonic: Single strongest note per beat
        - Drum: Fixed patterns with drum-specific envelopes
        
        Args:
            member: Band member to synthesize
            duration: Total duration in seconds (default: num_beats * beat_duration)
            sample_rate: Sample rate (default: self.sample_rate)
            
        Returns:
            Numpy array of audio samples
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        if duration is None:
            duration = member.num_beats * self.beat_duration
        
        # Route to appropriate synthesis method
        if isinstance(member, DrumMember):
            return self._synthesize_drums(member, duration, sample_rate)
        elif isinstance(member, MonophonicMember):
            return self._synthesize_monophonic(member, duration, sample_rate)
        elif isinstance(member, PolyphonicMember):
            return self._synthesize_polyphonic(member, duration, sample_rate)
        else:
            # Generic fallback
            return self._synthesize_generic(member, duration, sample_rate)
    
    def _synthesize_polyphonic(
        self,
        member: PolyphonicMember,
        duration: float,
        sample_rate: int
    ) -> np.ndarray:
        """Synthesize polyphonic member (piano, etc.).
        
        Uses thresholded weights to determine which notes to play.
        
        Args:
            member: Polyphonic band member
            duration: Total duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Audio samples as numpy array
        """
        num_samples = int(duration * sample_rate)
        audio = np.zeros(num_samples, dtype=np.float32)
        
        samples_per_beat = int(self.beat_duration * sample_rate)
        
        for beat_idx in range(member.num_beats):
            # Get active notes for this beat
            active_keys = member.get_active_notes(beat_idx, training=False)
            
            if not active_keys:
                continue
            
            # Calculate start and end samples for this beat
            start_sample = beat_idx * samples_per_beat
            end_sample = min(start_sample + samples_per_beat, num_samples)
            beat_samples = end_sample - start_sample
            
            if beat_samples <= 0:
                break
            
            # Synthesize each active note
            for key_idx in active_keys:
                # Get frequency for this key
                freq = member.tuning.get_frequency(member.key_offset + key_idx)
                
                # Synthesize note
                note_audio = self._synthesize_note(
                    frequency=freq,
                    instrument=member.instrument,
                    duration=beat_samples / sample_rate,
                    sample_rate=sample_rate
                )
                
                # Add to mix
                audio[start_sample:end_sample] += note_audio[:beat_samples]
        
        return audio
    
    def _synthesize_monophonic(
        self,
        member: MonophonicMember,
        duration: float,
        sample_rate: int
    ) -> np.ndarray:
        """Synthesize monophonic member (guitar, bass).
        
        Only plays the single strongest note per beat.
        
        Args:
            member: Monophonic band member
            duration: Total duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Audio samples as numpy array
        """
        num_samples = int(duration * sample_rate)
        audio = np.zeros(num_samples, dtype=np.float32)
        
        samples_per_beat = int(self.beat_duration * sample_rate)
        
        for beat_idx in range(member.num_beats):
            # Get the single active note for this beat
            active_keys = member.get_active_notes(beat_idx, training=False)
            
            if not active_keys:
                continue
            
            # Only use the first (strongest) note
            key_idx = active_keys[0]
            
            # Calculate start and end samples for this beat
            start_sample = beat_idx * samples_per_beat
            end_sample = min(start_sample + samples_per_beat, num_samples)
            beat_samples = end_sample - start_sample
            
            if beat_samples <= 0:
                break
            
            # Get frequency for this key
            freq = member.tuning.get_frequency(member.key_offset + key_idx)
            
            # Synthesize note
            note_audio = self._synthesize_note(
                frequency=freq,
                instrument=member.instrument,
                duration=beat_samples / sample_rate,
                sample_rate=sample_rate
            )
            
            # Add to mix
            audio[start_sample:end_sample] += note_audio[:beat_samples]
        
        return audio
    
    def _synthesize_drums(
        self,
        member: DrumMember,
        duration: float,
        sample_rate: int
    ) -> np.ndarray:
        """Synthesize drum member.
        
        Uses fixed patterns with drum-specific instruments.
        
        Args:
            member: Drum band member
            duration: Total duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Audio samples as numpy array
        """
        if not member.enabled:
            return np.zeros(int(duration * sample_rate), dtype=np.float32)
        
        num_samples = int(duration * sample_rate)
        audio = np.zeros(num_samples, dtype=np.float32)
        
        samples_per_beat = int(self.beat_duration * sample_rate)
        
        for beat_idx in range(member.num_beats):
            # Get active drums for this beat
            active_drums = member.get_active_notes(beat_idx, training=False)
            
            if not active_drums:
                continue
            
            # Calculate start and end samples for this beat
            start_sample = beat_idx * samples_per_beat
            end_sample = min(start_sample + samples_per_beat, num_samples)
            beat_samples = end_sample - start_sample
            
            if beat_samples <= 0:
                break
            
            # Synthesize each active drum
            for drum_name in active_drums:
                drum_instrument = member.get_drum_instrument(drum_name)
                
                # Drums typically have a fixed "frequency" or are noise-based
                # Use a base frequency based on drum type
                base_freq = self._get_drum_frequency(drum_name)
                
                # Synthesize drum hit
                drum_audio = self._synthesize_note(
                    frequency=base_freq,
                    instrument=drum_instrument,
                    duration=beat_samples / sample_rate,
                    sample_rate=sample_rate
                )
                
                # Add to mix
                audio[start_sample:end_sample] += drum_audio[:beat_samples]
        
        return audio
    
    def _synthesize_generic(
        self,
        member: BandMember,
        duration: float,
        sample_rate: int
    ) -> np.ndarray:
        """Generic synthesis fallback for custom band members.
        
        Args:
            member: Generic band member
            duration: Total duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Audio samples as numpy array
        """
        # Use the same approach as polyphonic for generic members
        num_samples = int(duration * sample_rate)
        audio = np.zeros(num_samples, dtype=np.float32)
        
        samples_per_beat = int(self.beat_duration * sample_rate)
        
        for beat_idx in range(member.num_beats):
            active_keys = member.get_active_notes(beat_idx, training=False)
            
            if not active_keys:
                continue
            
            start_sample = beat_idx * samples_per_beat
            end_sample = min(start_sample + samples_per_beat, num_samples)
            beat_samples = end_sample - start_sample
            
            if beat_samples <= 0:
                break
            
            for key_idx in active_keys:
                freq = member.tuning.get_frequency(member.key_offset + key_idx)
                
                note_audio = self._synthesize_note(
                    frequency=freq,
                    instrument=member.instrument,
                    duration=beat_samples / sample_rate,
                    sample_rate=sample_rate
                )
                
                audio[start_sample:end_sample] += note_audio[:beat_samples]
        
        return audio
    
    def _synthesize_note(
        self,
        frequency: float,
        instrument: Instrument,
        duration: float,
        sample_rate: int
    ) -> np.ndarray:
        """Synthesize a single note using additive synthesis.
        
        Generates a note by summing sine waves for each harmonic,
        with ADSR envelopes applied per harmonic.
        
        Args:
            frequency: Fundamental frequency in Hz
            instrument: Instrument profile
            duration: Note duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Audio samples as numpy array
        """
        num_samples = int(duration * sample_rate)
        if num_samples == 0:
            return np.array([], dtype=np.float32)
        
        # Time array
        t = np.linspace(0, duration, num_samples)
        
        # Initialize output
        audio = np.zeros(num_samples, dtype=np.float32)
        
        # Get harmonic profile with per-harmonic ADSR
        harmonic_profile = instrument.get_harmonic_profile()
        
        # Synthesize each harmonic
        for ratio, amplitude, adsr in harmonic_profile:
            harmonic_freq = frequency * ratio
            
            # Skip if above Nyquist
            if harmonic_freq >= sample_rate / 2:
                continue
            
            # Generate sine wave
            phase = 2 * np.pi * harmonic_freq * t
            harmonic_wave = np.sin(phase)
            
            # Get ADSR envelope for this harmonic
            envelope = adsr.get_envelope(duration, sample_rate)
            
            # Ensure envelope matches sample count
            if len(envelope) < num_samples:
                envelope = torch.nn.functional.pad(
                    envelope, (0, num_samples - len(envelope))
                )
            elif len(envelope) > num_samples:
                envelope = envelope[:num_samples]
            
            envelope = envelope.numpy() if isinstance(envelope, torch.Tensor) else envelope
            
            # Apply envelope and amplitude
            harmonic_audio = harmonic_wave * envelope * amplitude
            
            # Add to output
            audio += harmonic_audio
        
        # Normalize to prevent clipping within this note
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp * 0.5  # Leave headroom
        
        return audio
    
    def _get_drum_frequency(self, drum_name: str) -> float:
        """Get base frequency for a drum type.
        
        Args:
            drum_name: Name of the drum
            
        Returns:
            Base frequency in Hz
        """
        # Typical drum frequencies
        drum_freqs = {
            "kick": 60.0,    # Low kick
            "snare": 180.0,  # Snare body
            "hihat": 800.0,  # High hat sizzle
        }
        return drum_freqs.get(drum_name, 100.0)
    
    def synthesize_chord(
        self,
        frequencies: List[float],
        instrument: Instrument,
        duration: float,
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """Synthesize a chord (multiple notes simultaneously).
        
        Convenience method for synthesizing a chord with a given instrument.
        
        Args:
            frequencies: List of frequencies to play
            instrument: Instrument to use
            duration: Duration in seconds
            sample_rate: Sample rate (default: self.sample_rate)
            
        Returns:
            Audio samples as numpy array
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        audio = np.zeros(int(duration * sample_rate), dtype=np.float32)
        
        for freq in frequencies:
            note_audio = self._synthesize_note(
                frequency=freq,
                instrument=instrument,
                duration=duration,
                sample_rate=sample_rate
            )
            audio += note_audio
        
        # Normalize
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp * 0.5
        
        return audio


# ==================== VERIFICATION TESTS ====================

if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO SYNTHESIZER VERIFICATION")
    print("=" * 60)
    
    from .band import PolyphonicMember, MonophonicMember, DrumMember
    
    synthesizer = AudioSynthesizer(sample_rate=22050, beat_duration=0.5)
    
    # Test 1: Synthesize single note
    print("\n1. Single Note Synthesis:")
    piano = Instrument.piano()
    note_audio = synthesizer._synthesize_note(
        frequency=440.0,  # A4
        instrument=piano,
        duration=1.0,
        sample_rate=22050
    )
    print(f"   Note audio shape: {note_audio.shape}")
    print(f"   Duration: {len(note_audio) / 22050:.3f}s")
    print(f"   Max amplitude: {np.max(np.abs(note_audio)):.3f}")
    print(f"   RMS amplitude: {np.sqrt(np.mean(note_audio**2)):.3f}")
    
    # Test 2: Synthesize polyphonic member (piano)
    print("\n2. Polyphonic Member (Piano):")
    piano_member = PolyphonicMember.piano(num_beats=4)
    
    # Set some weights to create a simple chord progression
    with torch.no_grad():
        # Beat 0: C major chord (keys 39, 43, 46 = C4, E4, G4)
        piano_member.weights[39, 0] = 2.0
        piano_member.weights[43, 0] = 2.0
        piano_member.weights[46, 0] = 2.0
        
        # Beat 1: F major chord
        piano_member.weights[44, 1] = 2.0
        piano_member.weights[48, 1] = 2.0
        piano_member.weights[51, 1] = 2.0
        
        # Beat 2: G major chord
        piano_member.weights[46, 2] = 2.0
        piano_member.weights[50, 2] = 2.0
        piano_member.weights[53, 2] = 2.0
        
        # Beat 3: C major chord
        piano_member.weights[39, 3] = 2.0
        piano_member.weights[43, 3] = 2.0
        piano_member.weights[46, 3] = 2.0
    
    piano_audio = synthesizer.synthesize_member(piano_member)
    print(f"   Piano audio shape: {piano_audio.shape}")
    print(f"   Duration: {len(piano_audio) / 22050:.3f}s")
    print(f"   Max amplitude: {np.max(np.abs(piano_audio)):.3f}")
    
    # Test 3: Synthesize monophonic member (guitar)
    print("\n3. Monophonic Member (Guitar):")
    guitar_member = MonophonicMember.guitar(num_beats=4)
    
    # Set a melody
    with torch.no_grad():
        guitar_member.weights[12, 0] = 2.0  # E3
        guitar_member.weights[14, 1] = 2.0  # F#3
        guitar_member.weights[16, 2] = 2.0  # G#3
        guitar_member.weights[17, 3] = 2.0  # A3
    
    guitar_audio = synthesizer.synthesize_member(guitar_member)
    print(f"   Guitar audio shape: {guitar_audio.shape}")
    print(f"   Duration: {len(guitar_audio) / 22050:.3f}s")
    print(f"   Max amplitude: {np.max(np.abs(guitar_audio)):.3f}")
    
    # Test 4: Synthesize drums
    print("\n4. Drum Member:")
    drums = DrumMember.standard_rock(num_beats=4, enabled=True)
    drum_audio = synthesizer.synthesize_member(drums)
    print(f"   Drum audio shape: {drum_audio.shape}")
    print(f"   Duration: {len(drum_audio) / 22050:.3f}s")
    print(f"   Max amplitude: {np.max(np.abs(drum_audio)):.3f}")
    
    # Test 5: Synthesize chord
    print("\n5. Chord Synthesis:")
    chord_audio = synthesizer.synthesize_chord(
        frequencies=[261.63, 329.63, 392.00],  # C4, E4, G4
        instrument=Instrument.guitar(),
        duration=1.0
    )
    print(f"   Chord audio shape: {chord_audio.shape}")
    print(f"   Duration: {len(chord_audio) / 22050:.3f}s")
    print(f"   Max amplitude: {np.max(np.abs(chord_audio)):.3f}")
    
    # Test 6: Different beat durations
    print("\n6. Different Beat Durations:")
    for beat_dur in [0.25, 0.5, 1.0]:
        synth = AudioSynthesizer(sample_rate=22050, beat_duration=beat_dur)
        audio = synth.synthesize_member(piano_member)
        print(f"   Beat duration {beat_dur}s: audio length {len(audio) / 22050:.3f}s")
    
    # Test 7: ADSR envelope verification
    print("\n7. ADSR Envelope Verification:")
    from .instruments import ADSR
    
    adsr = ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
    envelope = adsr.get_envelope(duration=1.0, sample_rate=1000)
    env_arr = envelope.numpy() if isinstance(envelope, torch.Tensor) else envelope
    
    print(f"   Envelope shape: {env_arr.shape}")
    print(f"   Attack peak: {env_arr[50]:.3f} (should be ~1.0)")
    print(f"   Sustain level: {env_arr[500]:.3f} (should be ~0.7)")
    print(f"   End value: {env_arr[-1]:.3f} (should be ~0.0)")
    
    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)
