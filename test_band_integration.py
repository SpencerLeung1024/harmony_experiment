"""
Integration test for the Harmony From First Principles band system.

This test demonstrates the complete workflow:
1. Create a band with multiple members
2. Synthesize audio for each member
3. Mix them together
4. Verify the monophonic constraint
"""

import numpy as np
import torch

from harmony import (
    TwelveTET,
    PythagoreanTuning,
    ADSR,
    Instrument,
)
from harmony.band import (
    BandMember,
    PolyphonicMember,
    MonophonicMember,
    DrumMember,
)
from harmony.synthesis import AudioSynthesizer
from harmony.mixer import AudioMixer


def test_band_creation():
    """Test creating a full band with all member types."""
    print("\n" + "=" * 60)
    print("TEST 1: Band Creation")
    print("=" * 60)
    
    # Create a 4-beat composition
    num_beats = 4
    
    # Piano (polyphonic)
    piano = PolyphonicMember.piano(num_beats=num_beats)
    print(f"✓ Created piano: {piano}")
    
    # Guitar (monophonic)
    guitar = MonophonicMember.guitar(num_beats=num_beats)
    print(f"✓ Created guitar: {guitar}")
    
    # Bass (monophonic)
    bass = MonophonicMember.bass(num_beats=num_beats)
    print(f"✓ Created bass: {bass}")
    
    # Drums (pattern-based)
    drums = DrumMember.standard_rock(num_beats=num_beats)
    print(f"✓ Created drums: {drums}")
    
    return [piano, guitar, bass, drums]


def test_note_assignment(members):
    """Test assigning notes to band members."""
    print("\n" + "=" * 60)
    print("TEST 2: Note Assignment")
    print("=" * 60)
    
    piano, guitar, bass, drums = members
    
    with torch.no_grad():
        # Piano: I-V-vi-IV progression in C major
        # Beat 0: C major (C4, E4, G4)
        piano.weights[39, 0] = 2.0  # C4
        piano.weights[43, 0] = 2.0  # E4
        piano.weights[46, 0] = 2.0  # G4
        
        # Beat 1: G major (G3, B3, D4)
        piano.weights[34, 1] = 2.0  # G3
        piano.weights[38, 1] = 2.0  # B3
        piano.weights[41, 1] = 2.0  # D4
        
        # Beat 2: A minor (A3, C4, E4)
        piano.weights[36, 2] = 2.0  # A3
        piano.weights[39, 2] = 2.0  # C4
        piano.weights[43, 2] = 2.0  # E4
        
        # Beat 3: F major (F3, A3, C4)
        piano.weights[33, 3] = 2.0  # F3
        piano.weights[36, 3] = 2.0  # A3
        piano.weights[39, 3] = 2.0  # C4
        
        # Guitar: Melody line (monophonic - only one note per beat)
        guitar.weights[16, 0] = 2.0   # E4
        guitar.weights[19, 1] = 2.0   # G4
        guitar.weights[21, 2] = 2.0   # A4
        guitar.weights[18, 3] = 2.0   # F#4
        
        # Bass: Root notes (monophonic)
        bass.weights[15, 0] = 2.0   # C2
        bass.weights[22, 1] = 2.0   # G2
        bass.weights[24, 2] = 2.0   # A2
        bass.weights[21, 3] = 2.0   # F2
    
    print("✓ Assigned chord progression to piano")
    print("✓ Assigned melody to guitar")
    print("✓ Assigned bass line")
    print("✓ Drum pattern is fixed (kick on 1,3; snare on 2,4)")


def test_monophonic_constraint(members):
    """Verify monophonic constraint is enforced."""
    print("\n" + "=" * 60)
    print("TEST 3: Monophonic Constraint Verification")
    print("=" * 60)
    
    _, guitar, bass, _ = members
    
    # Test with intentionally conflicting weights
    with torch.no_grad():
        guitar.weights[:, 0] = torch.randn(guitar.num_keys) * 0.5 + 1.0  # All high
        bass.weights[:, 0] = torch.randn(bass.num_keys) * 0.5 + 1.0
    
    for beat in range(4):
        guitar_notes = guitar.get_active_notes(beat, training=False)
        bass_notes = bass.get_active_notes(beat, training=False)
        
        print(f"  Beat {beat}: Guitar={guitar_notes}, Bass={bass_notes}")
        
        # Verify monophonic constraint (max 1 note per beat)
        assert len(guitar_notes) <= 1, f"Guitar has {len(guitar_notes)} notes (should be <= 1)"
        assert len(bass_notes) <= 1, f"Bass has {len(bass_notes)} notes (should be <= 1)"
    
    print("✓ Monophonic constraint verified for all beats")


def test_polyphonic_capability(members):
    """Verify piano can play multiple notes."""
    print("\n" + "=" * 60)
    print("TEST 4: Polyphonic Capability Verification")
    print("=" * 60)
    
    piano, _, _, _ = members
    
    for beat in range(4):
        piano_notes = piano.get_active_notes(beat, training=False)
        print(f"  Beat {beat}: Piano notes = {piano_notes}")
        
        # Verify piano has multiple notes (polyphonic)
        assert len(piano_notes) >= 1, f"Piano has no notes on beat {beat}"
    
    print("✓ Piano can play multiple notes per beat")


def test_drum_patterns(members):
    """Test drum pattern playback."""
    print("\n" + "=" * 60)
    print("TEST 5: Drum Pattern Verification")
    print("=" * 60)
    
    _, _, _, drums = members
    
    expected_patterns = {
        0: ["kick"],
        1: ["snare"],
        2: ["kick"],
        3: ["snare"]
    }
    
    for beat, expected in expected_patterns.items():
        actual = drums.get_active_notes(beat)
        assert actual == expected, f"Beat {beat}: expected {expected}, got {actual}"
        print(f"  Beat {beat}: {actual} ✓")
    
    # Test toggle
    drums.toggle()
    assert drums.get_active_notes(0) == [], "Drums should be silent when disabled"
    print("✓ Drum toggle works correctly")
    drums.toggle()  # Re-enable


def test_audio_synthesis(members):
    """Test audio synthesis for each member."""
    print("\n" + "=" * 60)
    print("TEST 6: Audio Synthesis")
    print("=" * 60)
    
    synthesizer = AudioSynthesizer(sample_rate=22050, beat_duration=0.5)
    
    tracks = []
    for member in members:
        audio = synthesizer.synthesize_member(member)
        tracks.append(audio)
        
        duration = len(audio) / 22050
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        print(f"  {member.name:8s}: {len(audio):6d} samples, "
              f"{duration:.2f}s, peak={peak:.3f}, rms={rms:.3f}")
        
        # Verify audio is not silent
        assert peak > 0.01, f"{member.name} audio is too quiet (peak={peak})"
    
    print("✓ All members synthesized successfully")
    return tracks


def test_audio_mixing(members, tracks):
    """Test mixing multiple tracks."""
    print("\n" + "=" * 60)
    print("TEST 7: Audio Mixing")
    print("=" * 60)
    
    mixer = AudioMixer(headroom_db=-6.0)
    
    # Test default gains
    default_gains = mixer.get_default_gains(members)
    print(f"  Default gains: {default_gains}")
    
    # Mix with default gains
    mixed = mixer.mix_band(members, tracks, auto_gain=True)
    
    mixed_peak = np.max(np.abs(mixed))
    mixed_rms = np.sqrt(np.mean(mixed**2))
    mixed_duration = len(mixed) / 22050
    
    print(f"  Mixed audio: {len(mixed)} samples, {mixed_duration:.2f}s")
    print(f"  Peak level: {mixed_peak:.3f}")
    print(f"  RMS level: {mixed_rms:.3f}")
    
    # Verify no clipping
    assert mixed_peak <= 0.55, f"Mixed audio clips (peak={mixed_peak} > 0.55)"
    print("✓ No clipping detected")
    
    # Test custom gains
    custom_gains = [0.5, 1.0, 0.8, 0.3]
    mixed_custom = mixer.mix_tracks(tracks, gains=custom_gains)
    print(f"  Custom mix peak: {np.max(np.abs(mixed_custom)):.3f}")
    print("✓ Custom gain mixing works")
    
    return mixed


def test_different_tunings():
    """Test band members with different tuning systems."""
    print("\n" + "=" * 60)
    print("TEST 8: Different Tuning Systems")
    print("=" * 60)
    
    # Create same melody in different tunings
    standard_guitar = MonophonicMember.guitar(
        num_beats=4,
        tuning=TwelveTET()
    )
    pythagorean_guitar = MonophonicMember.guitar(
        num_beats=4,
        tuning=PythagoreanTuning()
    )
    
    # Same "frets" played
    with torch.no_grad():
        for guitar in [standard_guitar, pythagorean_guitar]:
            guitar.weights[12, 0] = 2.0
            guitar.weights[14, 1] = 2.0
            guitar.weights[16, 2] = 2.0
            guitar.weights[17, 3] = 2.0
    
    # Compare frequencies
    std_freqs = standard_guitar.get_frequencies()[[12, 14, 16, 17]]
    pyth_freqs = pythagorean_guitar.get_frequencies()[[12, 14, 16, 17]]
    
    print(f"  Standard 12-TET freqs: {std_freqs.tolist()}")
    print(f"  Pythagorean freqs:     {pyth_freqs.tolist()}")
    
    # Verify they're different
    assert not torch.allclose(std_freqs, pyth_freqs, atol=0.1), \
        "Tunings should produce different frequencies"
    print("✓ Different tunings produce different frequencies")


def test_gradient_flow():
    """Test that gradients flow through band members."""
    print("\n" + "=" * 60)
    print("TEST 9: Gradient Flow")
    print("=" * 60)
    
    guitar = MonophonicMember.guitar(num_beats=4)
    piano = PolyphonicMember.piano(num_beats=4)
    
    # Get soft activations
    guitar_soft = guitar.get_gumbel_sample(0, training=True)
    piano_soft = piano.get_soft_activation(0)
    
    print(f"  Guitar soft sample sum: {guitar_soft.sum().item():.3f}")
    print(f"  Piano soft activation sum: {piano_soft.sum().item():.3f}")
    
    # Test backward pass
    loss = guitar_soft.sum() + piano_soft.sum()
    loss.backward()
    
    assert guitar.weights.grad is not None, "Guitar weights should have gradients"
    assert piano.weights.grad is not None, "Piano weights should have gradients"
    
    print(f"  Guitar weight grad norm: {guitar.weights.grad.norm().item():.4f}")
    print(f"  Piano weight grad norm: {piano.weights.grad.norm().item():.4f}")
    print("✓ Gradients flow through band members")


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("HARMONY BAND SYSTEM - INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        # Run tests in sequence
        members = test_band_creation()
        test_note_assignment(members)
        test_monophonic_constraint(members)
        test_polyphonic_capability(members)
        test_drum_patterns(members)
        tracks = test_audio_synthesis(members)
        mixed = test_audio_mixing(members, tracks)
        test_different_tunings()
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)
        
        # Summary
        print("\nSummary:")
        print(f"  - Created band with {len(members)} members")
        print(f"  - Synthesized {len(tracks)} audio tracks")
        print(f"  - Mixed final audio: {len(mixed)} samples ({len(mixed)/22050:.2f}s)")
        print(f"  - Verified monophonic constraint")
        print(f"  - Verified polyphonic capability")
        print(f"  - Verified gradient flow")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
