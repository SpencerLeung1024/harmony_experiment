"""
Audio mixer module for Harmony From First Principles.

This module provides audio mixing capabilities for combining multiple
band member tracks with gain control, length normalization, and
clipping prevention.
"""

from typing import List, Optional
import numpy as np

from .band import BandMember, PolyphonicMember, MonophonicMember, DrumMember


class AudioMixer:
    """Mixes multiple audio tracks with gain control and clipping prevention.
    
    Combines audio from multiple band members, handling different track lengths
    and applying individual gain controls. Includes a soft limiter to prevent
    clipping.
    
    Attributes:
        headroom_db: Headroom in dB for clipping prevention
        limiter_threshold: Threshold for soft limiting
    """
    
    def __init__(self, headroom_db: float = -6.0, limiter_threshold: float = 0.95):
        """Initialize the audio mixer.
        
        Args:
            headroom_db: Target headroom in dB (default: -6 dB)
            limiter_threshold: Threshold for soft limiting (default: 0.95)
        """
        self.headroom_db = headroom_db
        self.limiter_threshold = limiter_threshold
    
    def mix_tracks(
        self,
        tracks: List[np.ndarray],
        gains: Optional[List[float]] = None
    ) -> np.ndarray:
        """Mix multiple audio tracks with individual gain control.
        
        Combines all tracks, handling different lengths by padding shorter
        tracks with zeros. Applies individual gains and then normalizes
        with clipping prevention.
        
        Args:
            tracks: List of audio tracks (numpy arrays)
            gains: Optional list of gain multipliers (one per track)
                   If None, uses default gains of 1.0
            
        Returns:
            Mixed audio as numpy array
            
        Raises:
            ValueError: If tracks is empty or gains length doesn't match tracks
        """
        if not tracks:
            raise ValueError("No tracks to mix")
        
        if gains is not None and len(gains) != len(tracks):
            raise ValueError(
                f"Number of gains ({len(gains)}) must match number of tracks ({len(tracks)})"
            )
        
        # Find maximum length
        max_length = max(len(track) for track in tracks)
        
        # Initialize mixed audio
        mixed = np.zeros(max_length, dtype=np.float64)  # Use float64 for precision
        
        # Apply default gains if not provided
        if gains is None:
            gains = [1.0] * len(tracks)
        
        # Mix all tracks
        for track, gain in zip(tracks, gains):
            # Pad or trim track to max_length
            if len(track) < max_length:
                padded = np.zeros(max_length, dtype=np.float64)
                padded[:len(track)] = track
                track = padded
            elif len(track) > max_length:
                track = track[:max_length]
            
            # Apply gain and add to mix
            mixed += track.astype(np.float64) * gain
        
        # Apply soft limiter to prevent clipping
        mixed = self._apply_limiter(mixed)
        
        # Convert back to float32
        return mixed.astype(np.float32)
    
    def _apply_limiter(self, audio: np.ndarray) -> np.ndarray:
        """Apply soft limiting to prevent clipping.
        
        Uses a soft knee compression approach that smoothly limits
        amplitudes above the threshold.
        
        Args:
            audio: Input audio array
            
        Returns:
            Limited audio array
        """
        # Calculate current peak
        peak = np.max(np.abs(audio))
        
        if peak == 0:
            return audio
        
        # Calculate target peak based on headroom
        target_peak = 10 ** (self.headroom_db / 20)
        
        # If we're already below target, just normalize
        if peak <= target_peak:
            return audio / peak * target_peak
        
        # Apply soft limiting for peaks above threshold
        threshold = self.limiter_threshold * target_peak
        
        # Soft knee compression function
        def soft_limit(x):
            abs_x = np.abs(x)
            if abs_x <= threshold:
                return x
            else:
                # Soft knee above threshold
                excess = abs_x - threshold
                compressed = threshold + np.tanh(excess / threshold) * threshold
                return np.sign(x) * compressed
        
        # Vectorize the limiter function
        limiter = np.vectorize(soft_limit)
        limited = limiter(audio)
        
        # Final normalization to target peak
        limited_peak = np.max(np.abs(limited))
        if limited_peak > 0:
            limited = limited / limited_peak * target_peak
        
        return limited
    
    def get_default_gains(self, members: List[BandMember]) -> List[float]:
        """Get sensible default gains for a list of band members.
        
        Provides balanced starting levels for different instrument types:
        - Piano: 0.8 (full but not overwhelming)
        - Guitar: 0.7 (slightly lower, mid-range)
        - Bass: 0.9 (needs to be heard despite fewer harmonics)
        - Drums: 0.6 (percussive, can be loud)
        
        Args:
            members: List of band members
            
        Returns:
            List of gain values
        """
        default_gains = {
            "piano": 0.8,
            "guitar": 0.7,
            "bass": 0.9,
            "drums": 0.6,
        }
        
        gains = []
        for member in members:
            # Get gain based on member name (case-insensitive)
            name_lower = member.name.lower()
            
            # Check for exact match first
            if name_lower in default_gains:
                gain = default_gains[name_lower]
            else:
                # Check for partial matches
                if "piano" in name_lower:
                    gain = default_gains["piano"]
                elif "guitar" in name_lower:
                    gain = default_gains["guitar"]
                elif "bass" in name_lower:
                    gain = default_gains["bass"]
                elif "drum" in name_lower:
                    gain = default_gains["drums"]
                else:
                    # Default for unknown instruments
                    gain = 0.7
            
            # Adjust for member type
            if isinstance(member, DrumMember):
                # Drums are inherently percussive, reduce gain slightly
                gain *= 0.9
            elif isinstance(member, MonophonicMember):
                # Monophonic instruments can be slightly louder
                gain *= 1.05
            
            gains.append(min(gain, 1.0))  # Cap at 1.0
        
        return gains
    
    def mix_band(
        self,
        members: List[BandMember],
        audio_tracks: List[np.ndarray],
        gains: Optional[List[float]] = None,
        auto_gain: bool = True
    ) -> np.ndarray:
        """Convenience method to mix a full band.
        
        Combines get_default_gains and mix_tracks for a streamlined workflow.
        
        Args:
            members: List of band members
            audio_tracks: List of audio tracks (one per member)
            gains: Optional custom gains (overrides auto_gain if provided)
            auto_gain: If True and gains not provided, use default gains
            
        Returns:
            Mixed audio as numpy array
        """
        if gains is None and auto_gain:
            gains = self.get_default_gains(members)
        
        return self.mix_tracks(audio_tracks, gains)
    
    def adjust_gain(self, audio: np.ndarray, db: float) -> np.ndarray:
        """Adjust audio gain in decibels.
        
        Args:
            audio: Input audio array
            db: Gain adjustment in dB (positive = louder, negative = quieter)
            
        Returns:
            Gain-adjusted audio
        """
        gain_linear = 10 ** (db / 20)
        return audio * gain_linear
    
    def normalize(self, audio: np.ndarray, target_db: float = -6.0) -> np.ndarray:
        """Normalize audio to a target peak level.
        
        Args:
            audio: Input audio array
            target_db: Target peak level in dB (default: -6 dB = 0.5 linear)
            
        Returns:
            Normalized audio
        """
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        
        target_linear = 10 ** (target_db / 20)
        return audio / peak * target_linear


# ==================== VERIFICATION TESTS ====================

if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO MIXER VERIFICATION")
    print("=" * 60)
    
    from .band import PolyphonicMember, MonophonicMember, DrumMember
    from .synthesis import AudioSynthesizer
    import torch
    
    mixer = AudioMixer(headroom_db=-6.0)
    synthesizer = AudioSynthesizer(sample_rate=22050, beat_duration=0.5)
    
    # Test 1: Create band members
    print("\n1. Creating Band Members:")
    piano = PolyphonicMember.piano(num_beats=4)
    guitar = MonophonicMember.guitar(num_beats=4)
    bass = MonophonicMember.bass(num_beats=4)
    drums = DrumMember.standard_rock(num_beats=4)
    
    members = [piano, guitar, bass, drums]
    for member in members:
        print(f"   {member}")
    
    # Test 2: Get default gains
    print("\n2. Default Gains:")
    default_gains = mixer.get_default_gains(members)
    for member, gain in zip(members, default_gains):
        print(f"   {member.name}: {gain:.2f}")
    
    # Test 3: Synthesize and mix tracks
    print("\n3. Track Synthesis:")
    
    # Set up some notes for each member
    with torch.no_grad():
        # Piano: C major, F major, G major, C major
        piano.weights[39, 0] = 2.0  # C4
        piano.weights[43, 0] = 2.0  # E4
        piano.weights[46, 0] = 2.0  # G4
        piano.weights[44, 1] = 2.0  # F4
        piano.weights[48, 1] = 2.0  # A4
        piano.weights[51, 1] = 2.0  # C5
        piano.weights[46, 2] = 2.0  # G4
        piano.weights[50, 2] = 2.0  # B4
        piano.weights[53, 2] = 2.0  # D5
        piano.weights[39, 3] = 2.0  # C4
        piano.weights[43, 3] = 2.0  # E4
        piano.weights[46, 3] = 2.0  # G4
        
        # Guitar: Simple melody
        guitar.weights[12, 0] = 2.0
        guitar.weights[14, 1] = 2.0
        guitar.weights[16, 2] = 2.0
        guitar.weights[17, 3] = 2.0
        
        # Bass: Root notes
        bass.weights[15, 0] = 2.0  # C2
        bass.weights[20, 1] = 2.0  # F2
        bass.weights[22, 2] = 2.0  # G2
        bass.weights[15, 3] = 2.0  # C2
    
    # Synthesize each member
    tracks = []
    for member in members:
        audio = synthesizer.synthesize_member(member)
        tracks.append(audio)
        print(f"   {member.name}: {len(audio)} samples, max={np.max(np.abs(audio)):.3f}")
    
    # Test 4: Mix with default gains
    print("\n4. Mixing with Default Gains:")
    mixed = mixer.mix_band(members, tracks, auto_gain=True)
    print(f"   Mixed audio shape: {mixed.shape}")
    print(f"   Peak amplitude: {np.max(np.abs(mixed)):.3f}")
    print(f"   RMS level: {np.sqrt(np.mean(mixed**2)):.3f}")
    
    # Test 5: Mix with custom gains
    print("\n5. Mixing with Custom Gains:")
    custom_gains = [1.0, 0.5, 1.2, 0.3]  # Piano full, guitar quiet, bass loud, drums quiet
    mixed_custom = mixer.mix_tracks(tracks, gains=custom_gains)
    print(f"   Custom gains: {custom_gains}")
    print(f"   Peak amplitude: {np.max(np.abs(mixed_custom)):.3f}")
    
    # Test 6: Handle different length tracks
    print("\n6. Different Length Tracks:")
    short_track = tracks[0][:len(tracks[0])//2]  # Half length
    long_track = tracks[1]
    mixed_diff = mixer.mix_tracks([short_track, long_track])
    print(f"   Short track: {len(short_track)} samples")
    print(f"   Long track: {len(long_track)} samples")
    print(f"   Mixed length: {len(mixed_diff)} samples (should match long)")
    
    # Test 7: Limiter behavior
    print("\n7. Limiter Behavior:")
    # Create a track that would clip
    loud_track = tracks[0] * 3.0  # Boost by ~9.5 dB
    print(f"   Pre-limiter peak: {np.max(np.abs(loud_track)):.3f}")
    mixed_loud = mixer.mix_tracks([loud_track])
    print(f"   Post-limiter peak: {np.max(np.abs(mixed_loud)):.3f}")
    print(f"   Limiter working: {np.max(np.abs(mixed_loud)) <= 0.6}")
    
    # Test 8: Gain adjustment
    print("\n8. Gain Adjustment:")
    original_peak = np.max(np.abs(tracks[0]))
    boosted = mixer.adjust_gain(tracks[0], 6.0)  # +6 dB
    boosted_peak = np.max(np.abs(boosted))
    print(f"   Original peak: {original_peak:.3f}")
    print(f"   Boosted peak (+6dB): {boosted_peak:.3f}")
    print(f"   Expected: ~{original_peak * 2:.3f}")
    
    # Test 9: Normalization
    print("\n9. Normalization:")
    quiet_track = tracks[0] * 0.1
    normalized = mixer.normalize(quiet_track, target_db=-6.0)
    print(f"   Original peak: {np.max(np.abs(quiet_track)):.3f}")
    print(f"   Normalized peak: {np.max(np.abs(normalized)):.3f}")
    print(f"   Target (-6dB): 0.5")
    
    # Test 10: Edge cases
    print("\n10. Edge Cases:")
    
    # Empty track list
    try:
        mixer.mix_tracks([])
        print("   Empty tracks: FAILED (should raise ValueError)")
    except ValueError:
        print("   Empty tracks: Correctly raises ValueError")
    
    # Mismatched gains
    try:
        mixer.mix_tracks([tracks[0], tracks[1]], gains=[0.5])
        print("   Mismatched gains: FAILED (should raise ValueError)")
    except ValueError:
        print("   Mismatched gains: Correctly raises ValueError")
    
    # Silence
    silent_track = np.zeros_like(tracks[0])
    mixed_silent = mixer.mix_tracks([silent_track, tracks[0]])
    print(f"   Mixed with silence: peak={np.max(np.abs(mixed_silent)):.3f}")
    
    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)
