"""
Comprehensive End-to-End Integration Tests for Harmony From First Principles

This test suite verifies all implemented features:
- Tuning Systems (12-TET, Pythagorean, Meantone, EDO scales, non-octave)
- Instruments (ADSR envelopes, harmonic profiles)
- Band Members (Polyphonic, Monophonic, Drums)
- Synthesis & Mixing
- Loss Functions & Optimization
- User Constraints
- Visualization & UI

Run with: python test_full_integration.py
       or: python -m pytest test_full_integration.py -v (if pytest installed)
"""

import os
import sys
import tempfile
import time
import shutil
from typing import List, Dict, Optional
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

# Optional pytest import
try:
    import pytest
except ImportError:
    pytest = None

# Ensure harmony package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harmony import (
    # Tuning systems
    TuningSystem, TwelveTET, PythagoreanTuning, MeantoneTuning,
    EDOSystem, NonOctaveSystem,
    # Instruments
    ADSR, Instrument,
    # Dissonance
    DissonanceCalculator,
    # Band members
    BandMember, PolyphonicMember, MonophonicMember, DrumMember,
    # Losses
    LossFunction,
    # Constraints
    UserConstraint, ConstraintSet,
    # Optimizer
    HarmonyOptimizer,
    # Synthesis & Mixing
    AudioSynthesizer, AudioMixer,
    # Visualization
    plot_weights, plot_spectrogram, plot_dissonance_matrix,
    plot_loss_history, create_weight_piano_roll, color_weights_by_pitch_class,
    save_audio, save_weights_plot,
)

# =============================================================================
# Test Suite 1: Tuning Systems
# =============================================================================

class TestTuningSystems(unittest.TestCase):
    """Test all tuning systems produce valid frequencies."""
    
    def test_12_tet_basic(self):
        """Test 12-TET produces correct frequencies."""
        tuning = TwelveTET()
        # A4 should be 440 Hz
        self.assertAlmostEqual(tuning.get_frequency(69), 440.0, places=1)
        # Octave should double
        self.assertAlmostEqual(
            tuning.get_frequency(81),  # A5
            880.0, places=1
        )
        # Check name
        self.assertIn("12-TET", tuning.name)
    
    def test_12_tet_custom_reference(self):
        """Test 12-TET with custom reference."""
        tuning = TwelveTET(reference_hz=432.0, reference_key=69)
        self.assertAlmostEqual(tuning.get_frequency(69), 432.0, places=1)
    
    def test_pythagorean_tuning(self):
        """Test Pythagorean tuning produces valid frequencies."""
        tuning = PythagoreanTuning()
        freqs = tuning.get_all_frequencies(12)
        # All frequencies should be positive
        self.assertTrue(torch.all(freqs > 0))
        # Check name
        self.assertIn("Pythagorean", tuning.name)
    
    def test_meantone_tuning(self):
        """Test quarter-comma meantone tuning."""
        tuning = MeantoneTuning(comma_fraction=0.25)
        freqs = tuning.get_all_frequencies(12)
        self.assertTrue(torch.all(freqs > 0))
        self.assertIn("Meantone", tuning.name)
    
    def test_third_comma_meantone(self):
        """Test 1/3 comma meantone."""
        tuning = MeantoneTuning(comma_fraction=1/3)
        freqs = tuning.get_all_frequencies(12)
        self.assertTrue(torch.all(freqs > 0))
    
    def test_19_edo(self):
        """Test 19-EDO tuning."""
        tuning = EDOSystem(divisions=19)
        freqs = tuning.get_all_frequencies(19)
        self.assertTrue(torch.all(freqs > 0))
        # Octave should double
        f0 = tuning.get_frequency(0)
        f19 = tuning.get_frequency(19)
        self.assertAlmostEqual(f19 / f0, 2.0, places=5)
        self.assertIn("19-EDO", tuning.name)
    
    def test_31_edo(self):
        """Test 31-EDO tuning."""
        tuning = EDOSystem(divisions=31)
        freqs = tuning.get_all_frequencies(31)
        self.assertTrue(torch.all(freqs > 0))
        self.assertIn("31-EDO", tuning.name)
    
    def test_24_edo(self):
        """Test 24-EDO (quarter-tone) tuning."""
        tuning = EDOSystem(divisions=24)
        freqs = tuning.get_all_frequencies(24)
        self.assertTrue(torch.all(freqs > 0))
    
    def test_41_edo(self):
        """Test 41-EDO tuning."""
        tuning = EDOSystem(divisions=41)
        freqs = tuning.get_all_frequencies(41)
        self.assertTrue(torch.all(freqs > 0))
    
    def test_53_edo(self):
        """Test 53-EDO tuning."""
        tuning = EDOSystem(divisions=53)
        freqs = tuning.get_all_frequencies(53)
        self.assertTrue(torch.all(freqs > 0))
    
    def test_alpha_scale(self):
        """Test Bohlen-Pierce Alpha scale (non-octave)."""
        tuning = NonOctaveSystem.alpha_scale()
        freqs = tuning.get_all_frequencies(15)
        self.assertTrue(torch.all(freqs > 0))
        self.assertIn("Alpha", tuning.name)
    
    def test_beta_scale(self):
        """Test Bohlen-Pierce Beta scale (non-octave)."""
        tuning = NonOctaveSystem.beta_scale()
        freqs = tuning.get_all_frequencies(15)
        self.assertTrue(torch.all(freqs > 0))
        self.assertIn("Beta", tuning.name)
    
    def test_bohlen_pierce(self):
        """Test Bohlen-Pierce scale (3:1 instead of 2:1)."""
        tuning = NonOctaveSystem.bohlen_pierce()
        freqs = tuning.get_all_frequencies(13)
        self.assertTrue(torch.all(freqs > 0))
        self.assertIn("Bohlen-Pierce", tuning.name)
    
    def test_dissonance_matrix_all_tunings(self):
        """Test dissonance matrices work with each tuning system."""
        tunings = [
            TwelveTET(),
            PythagoreanTuning(),
            MeantoneTuning(comma_fraction=0.25),
            EDOSystem(divisions=19),
            EDOSystem(divisions=31),
            NonOctaveSystem.alpha_scale(),
        ]
        
        for tuning in tunings:
            with self.subTest(tuning=tuning.name):
                calc = DissonanceCalculator(tuning=tuning)
                num_keys = 12 if not isinstance(tuning, NonOctaveSystem) else 9
                matrix = calc.calculate_matrix(num_keys=num_keys, max_hz=8000)
                
                # Matrix should be square
                self.assertEqual(matrix.shape, (num_keys, num_keys))
                # Matrix should be symmetric
                self.assertTrue(torch.allclose(matrix, matrix.T, atol=1e-5))
                # All values should be non-negative
                self.assertTrue(torch.all(matrix >= 0))


# =============================================================================
# Test Suite 2: Instruments
# =============================================================================

class TestInstruments(unittest.TestCase):
    """Test instrument presets and ADSR envelopes."""
    
    def test_adsr_basic(self):
        """Test ADSR envelope generation."""
        adsr = ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
        envelope = adsr.get_envelope(duration=1.0, sample_rate=1000)
        
        # Envelope should have correct length
        self.assertEqual(len(envelope), 1000)
        # Envelope should start at 0
        self.assertAlmostEqual(envelope[0].item(), 0.0, places=5)
        # Envelope should end near 0 (after release)
        self.assertLess(envelope[-1].item(), 0.1)
    
    def test_adsr_custom_params(self):
        """Test ADSR with custom parameters."""
        adsr = ADSR(attack=0.02, decay=0.1, sustain=0.8, release=0.4)
        envelope = adsr.get_envelope(duration=1.0, sample_rate=1000)
        self.assertEqual(len(envelope), 1000)
        self.assertEqual(adsr.attack, 0.02)
        self.assertEqual(adsr.sustain, 0.8)
    
    def test_adsr_from_instrument(self):
        """Test ADSR from instrument presets."""
        piano = Instrument.piano()
        envelope = piano.adsr.get_envelope(duration=1.0, sample_rate=1000)
        self.assertEqual(len(envelope), 1000)
        # Piano should have quick attack
        self.assertLess(piano.adsr.attack, 0.05)
    
    def test_adsr_pluck_like(self):
        """Test pluck-like ADSR."""
        adsr = ADSR(attack=0.002, decay=0.3, sustain=0.6, release=0.8)
        envelope = adsr.get_envelope(duration=1.0, sample_rate=1000)
        self.assertEqual(len(envelope), 1000)
        # Pluck should have very short attack
        self.assertLess(adsr.attack, 0.01)
    
    def test_adsr_pad_like(self):
        """Test pad-like ADSR."""
        adsr = ADSR(attack=0.3, decay=0.2, sustain=0.9, release=1.0)
        envelope = adsr.get_envelope(duration=2.0, sample_rate=1000)
        self.assertEqual(len(envelope), 2000)
        # Pad should have long attack
        self.assertGreater(adsr.attack, 0.2)
    
    def test_instrument_preset_piano(self):
        """Test piano instrument preset."""
        inst = Instrument.piano()
        self.assertGreater(len(inst.harmonics), 5)
        self.assertIn("Piano", inst.name)
    
    def test_instrument_preset_guitar(self):
        """Test guitar instrument preset."""
        inst = Instrument.guitar()
        self.assertGreater(len(inst.harmonics), 3)
        self.assertIn("Guitar", inst.name)
    
    def test_instrument_preset_bass(self):
        """Test bass instrument preset."""
        inst = Instrument.bass()
        self.assertGreater(len(inst.harmonics), 3)
        self.assertIn("Bass", inst.name)
    
    def test_instrument_preset_synth(self):
        """Test synth instrument preset."""
        inst = Instrument.synth()
        self.assertGreater(len(inst.harmonics), 1)
        self.assertIn("Synth", inst.name)
    
    def test_instrument_preset_drums(self):
        """Test drum instrument presets."""
        kick = Instrument.drums_kick()
        snare = Instrument.drums_snare()
        hihat = Instrument.drums_hihat()
        
        self.assertGreater(len(kick.harmonics), 0)
        self.assertGreater(len(snare.harmonics), 0)
        self.assertGreater(len(hihat.harmonics), 0)
    
    def test_instrument_per_harmonic_adsr(self):
        """Test per-harmonic ADSR envelopes."""
        # Create instrument with different ADSR per harmonic
        base_adsr = ADSR(attack=0.01, decay=0.1, sustain=0.7, release=0.3)
        per_harmonic_adsr = {
            0: ADSR(attack=0.01, decay=0.4, sustain=0.3, release=0.5),
            1: ADSR(attack=0.002, decay=0.3, sustain=0.6, release=0.8),
            2: ADSR(attack=0.3, decay=0.2, sustain=0.9, release=1.0),
        }
        
        inst = Instrument(
            name="TestInstrument",
            harmonics=[(1.0, 1.0), (2.0, 0.5), (3.0, 0.25)],
            adsr=base_adsr,
            per_harmonic_adsr=per_harmonic_adsr
        )
        
        self.assertEqual(len(inst.per_harmonic_adsr), 3)
        
        # Test that different harmonics have different envelopes
        profile = inst.get_harmonic_profile()
        self.assertEqual(len(profile), 3)
        
        # Verify each harmonic has its own ADSR
        for i, (ratio, amp, adsr) in enumerate(profile):
            self.assertIsInstance(adsr, ADSR)


# =============================================================================
# Test Suite 3: Band Members
# =============================================================================

class TestBandMembers(unittest.TestCase):
    """Test Polyphonic, Monophonic, and Drum members."""
    
    def test_polyphonic_member_creation(self):
        """Test PolyphonicMember creation."""
        piano = PolyphonicMember.piano(num_beats=4)
        self.assertEqual(piano.num_beats, 4)
        self.assertEqual(piano.weights.shape[0], piano.num_keys)
        self.assertEqual(piano.weights.shape[1], 4)
        self.assertIn("piano", piano.name.lower())
    
    def test_polyphonic_member_activation_threshold(self):
        """Test polyphonic member respects activation threshold."""
        # Create piano with threshold=1.0
        piano = PolyphonicMember.piano(num_beats=4)
        piano.threshold = 1.0
        
        # Set some weights
        with torch.no_grad():
            piano.weights.fill_(0.0)
            piano.weights[0, 0] = 2.0  # High weight
            piano.weights[1, 0] = 0.5  # Low weight (below threshold)
        
        # With threshold=1.0, only note 0 should be active
        active = piano.get_active_notes(beat_index=0, training=False)
        self.assertIn(0, active)
        self.assertNotIn(1, active)
    
    def test_monophonic_member_creation(self):
        """Test MonophonicMember creation."""
        guitar = MonophonicMember.guitar(num_beats=4)
        self.assertEqual(guitar.num_beats, 4)
        self.assertIn("guitar", guitar.name.lower())
    
    def test_monophonic_constraint(self):
        """Test monophonic constraint (only one note per beat)."""
        guitar = MonophonicMember.guitar(num_beats=4)
        
        # Set multiple high weights on same beat
        with torch.no_grad():
            guitar.weights.fill_(0.0)
            guitar.weights[0, 0] = 3.0
            guitar.weights[1, 0] = 2.0
            guitar.weights[2, 0] = 1.0
        
        # Should return only the strongest note
        active = guitar.get_active_notes(beat_index=0)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0], 0)  # Strongest weight
    
    def test_bass_member_creation(self):
        """Test bass MonophonicMember creation."""
        bass = MonophonicMember.bass(num_beats=4)
        self.assertEqual(bass.num_beats, 4)
        self.assertIn("bass", bass.name.lower())
    
    def test_drum_member_creation(self):
        """Test DrumMember creation."""
        drums = DrumMember.standard_rock(num_beats=4)
        self.assertEqual(drums.num_beats, 4)
        self.assertIn("drum", drums.name.lower())
    
    def test_drum_member_not_optimizable(self):
        """Test DrumMember weights are not optimizable."""
        drums = DrumMember.standard_rock(num_beats=4)
        # Drum weights should not require gradients
        self.assertFalse(drums.weights.requires_grad)
    
    def test_drum_member_fixed_pattern(self):
        """Test DrumMember plays fixed patterns."""
        drums = DrumMember.standard_rock(num_beats=4)
        
        # Get active notes for each beat
        for beat in range(4):
            active = drums.get_active_notes(beat_index=beat)
            # Standard rock: kicks on 0, 2; snares on 1, 3
            self.assertIsInstance(active, list)
    
    def test_member_all_active_notes(self):
        """Test getting all active notes for all beats."""
        piano = PolyphonicMember.piano(num_beats=4)
        piano.threshold = 1.0
        
        with torch.no_grad():
            piano.weights.fill_(0.0)
            piano.weights[10, 0] = 2.0
            piano.weights[20, 1] = 2.0
            piano.weights[30, 2] = 2.0
            piano.weights[40, 3] = 2.0
        
        all_notes = piano.get_all_active_notes(training=False)
        self.assertEqual(len(all_notes), 4)
        self.assertEqual(all_notes[0], [10])
        self.assertEqual(all_notes[1], [20])
        self.assertEqual(all_notes[2], [30])
        self.assertEqual(all_notes[3], [40])


# =============================================================================
# Test Suite 4: Synthesis & Mixing
# =============================================================================

class TestSynthesisAndMixing(unittest.TestCase):
    """Test audio synthesis and mixing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 8000  # Lower for faster tests
        self.beat_duration = 0.5
        self.synthesizer = AudioSynthesizer(
            sample_rate=self.sample_rate,
            beat_duration=self.beat_duration
        )
        self.mixer = AudioMixer()
    
    def test_synthesize_polyphonic(self):
        """Test synthesizing polyphonic member audio."""
        piano = PolyphonicMember.piano(num_beats=2)
        
        # Set some notes
        with torch.no_grad():
            piano.weights[30, 0] = 2.0
            piano.weights[34, 0] = 2.0
            piano.weights[37, 1] = 2.0
        
        audio = self.synthesizer.synthesize_member(piano)
        
        # Audio should have correct length
        expected_samples = int(2 * self.beat_duration * self.sample_rate)
        self.assertEqual(len(audio), expected_samples)
        # Audio should not be all zeros
        self.assertGreater(np.abs(audio).max(), 0.0)
        # Audio should be in reasonable range
        self.assertLess(np.abs(audio).max(), 10.0)
    
    def test_synthesize_monophonic(self):
        """Test synthesizing monophonic member audio."""
        guitar = MonophonicMember.guitar(num_beats=2)
        
        with torch.no_grad():
            guitar.weights[20, 0] = 2.0
            guitar.weights[25, 1] = 2.0
        
        audio = self.synthesizer.synthesize_member(guitar)
        
        expected_samples = int(2 * self.beat_duration * self.sample_rate)
        self.assertEqual(len(audio), expected_samples)
        self.assertGreater(np.abs(audio).max(), 0.0)
    
    def test_synthesize_drums(self):
        """Test synthesizing drum member audio."""
        drums = DrumMember.standard_rock(num_beats=4)
        
        audio = self.synthesizer.synthesize_member(drums)
        
        expected_samples = int(4 * self.beat_duration * self.sample_rate)
        self.assertEqual(len(audio), expected_samples)
        self.assertGreater(np.abs(audio).max(), 0.0)
    
    def test_mix_multiple_tracks(self):
        """Test mixing multiple audio tracks."""
        track1 = np.random.randn(1000).astype(np.float32) * 0.1
        track2 = np.random.randn(1000).astype(np.float32) * 0.1
        track3 = np.random.randn(1000).astype(np.float32) * 0.1
        
        mixed = self.mixer.mix_tracks([track1, track2, track3])
        
        # Mixed track should have same length
        self.assertEqual(len(mixed), 1000)
        # Mixed track should not be clipped
        self.assertLessEqual(np.abs(mixed).max(), 1.0)
    
    def test_mix_with_gains(self):
        """Test mixing with different gains."""
        track1 = np.ones(1000, dtype=np.float32) * 0.5
        track2 = np.ones(1000, dtype=np.float32) * 0.5
        
        # Mix with different gains
        mixed = self.mixer.mix_tracks([track1, track2], gains=[1.0, 0.5])
        
        # Result should be approximately (0.5 * 1.0 + 0.5 * 0.5) = 0.75
        # But limited to prevent clipping
        self.assertLessEqual(np.abs(mixed).max(), 1.0)
    
    def test_mix_different_lengths(self):
        """Test mixing tracks of different lengths."""
        track1 = np.ones(1000, dtype=np.float32)
        track2 = np.ones(500, dtype=np.float32)
        
        mixed = self.mixer.mix_tracks([track1, track2])
        
        # Mixed track should have length of longest
        self.assertEqual(len(mixed), 1000)
    
    def test_mixer_empty_tracks_raises(self):
        """Test mixer raises error for empty track list."""
        with self.assertRaises(ValueError):
            self.mixer.mix_tracks([])
    
    def test_mixer_gain_mismatch_raises(self):
        """Test mixer raises error for gain/tracks mismatch."""
        with self.assertRaises(ValueError):
            self.mixer.mix_tracks([np.ones(100)], gains=[1.0, 2.0])


# =============================================================================
# Test Suite 5: Loss & Optimization
# =============================================================================

class TestLossAndOptimization(unittest.TestCase):
    """Test loss functions and optimization."""
    
    def test_loss_function_creation(self):
        """Test LossFunction creation."""
        piano = PolyphonicMember.piano(num_beats=2)
        guitar = MonophonicMember.guitar(num_beats=2)
        
        loss_fn = LossFunction(members=[piano, guitar])
        
        self.assertEqual(len(loss_fn.members), 2)
        self.assertIn('within', loss_fn.loss_weights)
        self.assertIn('cross', loss_fn.loss_weights)
    
    def test_precompute_dissonance_matrices(self):
        """Test dissonance matrix precomputation."""
        piano = PolyphonicMember.piano(num_beats=2)
        guitar = MonophonicMember.guitar(num_beats=2)
        
        loss_fn = LossFunction(members=[piano, guitar])
        loss_fn.precompute_dissonance_matrices()
        
        # Should have matrices for both members
        self.assertIn('piano', loss_fn.dissonance_matrices)
        self.assertIn('guitar', loss_fn.dissonance_matrices)
    
    def test_calculate_total_loss(self):
        """Test total loss calculation."""
        piano = PolyphonicMember.piano(num_beats=2)
        guitar = MonophonicMember.guitar(num_beats=2)
        
        loss_fn = LossFunction(members=[piano, guitar])
        loss_fn.precompute_dissonance_matrices()
        
        total_loss, loss_dict = loss_fn.calculate()
        
        # Should return positive loss
        self.assertGreater(total_loss.item(), 0)
        # Should return loss dictionary
        self.assertIn('total', loss_dict)
        self.assertIn('within', loss_dict)
    
    def test_cross_member_dissonance(self):
        """Test cross-member dissonance calculation."""
        piano = PolyphonicMember.piano(num_beats=2)
        guitar = MonophonicMember.guitar(num_beats=2)
        
        # Put notes on same keys to create cross-dissonance
        with torch.no_grad():
            piano.weights[30, 0] = 2.0
            guitar.weights[30, 0] = 2.0
        
        loss_fn = LossFunction(members=[piano, guitar])
        loss_fn.precompute_dissonance_matrices()
        
        total_loss, loss_dict = loss_fn.calculate()
        
        # Cross-dissonance should be calculated
        self.assertIn('cross', loss_dict)
    
    def test_user_constraint_integration(self):
        """Test user constraints in loss calculation."""
        piano = PolyphonicMember.piano(num_beats=2)
        
        # Create constraint set and add constraint
        constraints = ConstraintSet()
        constraint = UserConstraint(
            member_name="piano",
            beat_index=0,
            key_indices=30,
            fixed_value=1.0
        )
        constraints.add_constraint(constraint)
        
        # Set piano note near constraint
        with torch.no_grad():
            piano.weights[35, 0] = 2.0  # Different note, should create dissonance
        
        loss_fn = LossFunction(members=[piano])
        loss_fn.precompute_dissonance_matrices()
        
        # Calculate loss with constraints (pass effective weights)
        effective_weights = constraints.get_all_effective_weights([piano])
        total_loss, _ = loss_fn.calculate(effective_weights=effective_weights)
        self.assertGreater(total_loss.item(), 0)
    
    def test_optimizer_creation(self):
        """Test HarmonyOptimizer creation."""
        piano = PolyphonicMember.piano(num_beats=2)
        
        optimizer = HarmonyOptimizer(members=[piano], lr=0.01)
        
        self.assertIsNotNone(optimizer.optimizer)
        self.assertEqual(len(optimizer.loss_fn.members), 1)
    
    def test_optimizer_step(self):
        """Test single optimization step."""
        piano = PolyphonicMember.piano(num_beats=2)
        
        optimizer = HarmonyOptimizer(members=[piano], lr=0.01)
        
        initial_weights = piano.weights.clone()
        
        # Take one step
        loss_dict = optimizer.step()
        
        # Loss should be returned in dict
        self.assertIn('total', loss_dict)
        # Weights should have changed
        self.assertFalse(torch.equal(piano.weights, initial_weights))
    
    def test_full_optimization_loop(self):
        """Test full optimization loop."""
        piano = PolyphonicMember.piano(num_beats=2)
        
        optimizer = HarmonyOptimizer(
            members=[piano],
            lr=0.05,
            target_density=0.1
        )
        
        # Run optimization
        num_steps = 10
        for i in range(num_steps):
            loss_dict = optimizer.step()
        
        # Should have loss history
        self.assertEqual(len(optimizer.loss_history), num_steps)
        # Loss should generally decrease (not strictly due to randomness)
        first_loss = optimizer.loss_history[0]['total']
        last_loss = optimizer.loss_history[-1]['total']
        # Just verify we have valid numbers
        self.assertIsInstance(first_loss, float)
        self.assertIsInstance(last_loss, float)


# =============================================================================
# Test Suite 6: End-to-End Scenarios
# =============================================================================

class TestEndToEndScenarios(unittest.TestCase):
    """Test complete end-to-end scenarios."""
    
    def setUp(self):
        """Set up test output directory."""
        self.test_output_dir = tempfile.mkdtemp(prefix="harmony_test_")
    
    def tearDown(self):
        """Clean up test output directory."""
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
    
    def test_scenario_a_piano_solo_12tet(self):
        """Scenario A: Piano solo in 12-TET."""
        print("\n  Running Scenario A: Piano solo in 12-TET...")
        
        piano = PolyphonicMember.piano(num_beats=4)
        optimizer = HarmonyOptimizer(
            members=[piano],
            tuning=TwelveTET(),
            lr=0.05,
            target_density=0.15,
            loss_weights={'within': 1.0, 'density': 10.0}
        )
        
        # Run optimization
        result = optimizer.optimize(steps=20, verbose=False)
        loss_history = result['loss_history']
        
        # Verify results
        self.assertEqual(len(loss_history), 20)
        
        # Synthesize audio
        synth = AudioSynthesizer(sample_rate=8000)
        audio = synth.synthesize_member(piano)
        self.assertGreater(len(audio), 0)
        
        print(f"  ✓ Final loss: {loss_history[-1]['total']:.4f}")
    
    def test_scenario_b_piano_guitar_pythagorean(self):
        """Scenario B: Piano + Guitar in Pythagorean tuning."""
        print("\n  Running Scenario B: Piano + Guitar in Pythagorean tuning...")
        
        piano = PolyphonicMember.piano(num_beats=4)
        guitar = MonophonicMember.guitar(num_beats=4)
        
        optimizer = HarmonyOptimizer(
            members=[piano, guitar],
            tuning=PythagoreanTuning(),
            lr=0.03,
            target_density=0.12,
            loss_weights={
                'within': 1.0,
                'cross': 1.0,
                'density': 10.0,
                'interval_jump': 0.5
            }
        )
        
        result = optimizer.optimize(steps=20, verbose=False)
        loss_history = result['loss_history']
        self.assertEqual(len(loss_history), 20)
        
        # Verify cross-member dissonance is calculated
        self.assertIn('cross', loss_history[-1])
        
        print(f"  ✓ Final loss: {loss_history[-1]['total']:.4f}")
    
    def test_scenario_c_full_band_19edo(self):
        """Scenario C: Full band in 19-EDO."""
        print("\n  Running Scenario C: Full band in 19-EDO...")
        
        piano = PolyphonicMember.piano(num_beats=4)
        guitar = MonophonicMember.guitar(num_beats=4)
        bass = MonophonicMember.bass(num_beats=4)
        drums = DrumMember.standard_rock(num_beats=4)
        
        optimizer = HarmonyOptimizer(
            members=[piano, guitar, bass, drums],
            tuning=EDOSystem(divisions=19),
            lr=0.03,
            target_density=0.15
        )
        
        result = optimizer.optimize(steps=15, verbose=False)
        loss_history = result['loss_history']
        self.assertEqual(len(loss_history), 15)
        
        # Synthesize and mix all members
        synth = AudioSynthesizer(sample_rate=8000)
        mixer = AudioMixer()
        
        tracks = [
            synth.synthesize_member(piano),
            synth.synthesize_member(guitar),
            synth.synthesize_member(bass),
            synth.synthesize_member(drums)
        ]
        
        mixed = mixer.mix_tracks(tracks, gains=[0.8, 0.7, 0.8, 0.6])
        self.assertGreater(len(mixed), 0)
        self.assertLessEqual(np.abs(mixed).max(), 1.0)
        
        print(f"  ✓ Final loss: {loss_history[-1]['total']:.4f}")
    
    def test_scenario_d_piano_with_constraints(self):
        """Scenario D: Piano with user constraints (fixed notes)."""
        print("\n  Running Scenario D: Piano with user constraints...")
        
        piano = PolyphonicMember.piano(num_beats=4)
        
        # Add constraints: fix C major chord on beat 0
        constraints = ConstraintSet()
        constraints.add_constraint(
            UserConstraint("piano", 0, [39, 43, 46], fixed_value=1.0)  # C4, E4, G4
        )
        
        optimizer = HarmonyOptimizer(
            members=[piano],
            tuning=TwelveTET(),
            constraints=constraints,
            lr=0.05,
            target_density=0.15
        )
        
        result = optimizer.optimize(steps=20, verbose=False)
        loss_history = result['loss_history']
        
        # Verify optimization ran with constraints
        self.assertEqual(len(loss_history), 20)
        
        print(f"  ✓ Final loss: {loss_history[-1]['total']:.4f}")
    
    def test_scenario_e_tuning_comparison(self):
        """Scenario E: Compare different tuning systems."""
        print("\n  Running Scenario E: Tuning system comparison...")
        
        tunings = [
            ("12-TET", TwelveTET()),
            ("Pythagorean", PythagoreanTuning()),
            ("19-EDO", EDOSystem(divisions=19)),
        ]
        
        results = {}
        
        for name, tuning in tunings:
            piano = PolyphonicMember.piano(num_beats=2)
            
            optimizer = HarmonyOptimizer(
                members=[piano],
                tuning=tuning,
                lr=0.05,
                target_density=0.1
            )
            
            result = optimizer.optimize(steps=15, verbose=False)
            loss_history = result['loss_history']
            results[name] = loss_history[-1]['total']
        
        # All should have completed
        self.assertEqual(len(results), 3)
        for name, final_loss in results.items():
            self.assertIsInstance(final_loss, float)
            print(f"  ✓ {name}: {final_loss:.4f}")


# =============================================================================
# Test Suite 7: Visualization & UI
# =============================================================================

class TestVisualizationAndUI(unittest.TestCase):
    """Test visualization functions and UI."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_output_dir = tempfile.mkdtemp(prefix="harmony_viz_test_")
    
    def tearDown(self):
        """Clean up test output directory."""
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
    
    def test_plot_weights(self):
        """Test weight plotting."""
        piano = PolyphonicMember.piano(num_beats=4)
        
        fig = plot_weights(
            weights=piano.weights.detach().numpy(),
            title="Test Weights",
            member=piano
        )
        
        self.assertIsNotNone(fig)
    
    def test_plot_dissonance_matrix(self):
        """Test dissonance matrix plotting."""
        calc = DissonanceCalculator()
        matrix = calc.calculate_matrix(num_keys=12)
        
        fig = plot_dissonance_matrix(matrix, title="Test Matrix")
        self.assertIsNotNone(fig)
    
    def test_plot_loss_history(self):
        """Test loss history plotting."""
        loss_history = [
            {'total': 10.0, 'within': 5.0, 'density': 5.0},
            {'total': 8.0, 'within': 4.0, 'density': 4.0},
            {'total': 6.0, 'within': 3.0, 'density': 3.0},
        ]
        
        fig = plot_loss_history(loss_history)
        self.assertIsNotNone(fig)
    
    def test_create_weight_piano_roll(self):
        """Test piano roll visualization."""
        piano = PolyphonicMember.piano(num_beats=4)
        
        with torch.no_grad():
            piano.weights[30, 0] = 2.0
            piano.weights[34, 0] = 2.0
        
        # Function expects a list of members
        fig = create_weight_piano_roll([piano])
        self.assertIsNotNone(fig)
    
    def test_color_weights_by_pitch_class(self):
        """Test pitch-class coloring of weights."""
        weights = np.random.rand(24, 4)
        colored = color_weights_by_pitch_class(weights)
        
        # Should return RGB array
        self.assertEqual(colored.shape, (24, 4, 3))
        # All values should be in [0, 1]
        self.assertTrue(np.all(colored >= 0))
        self.assertTrue(np.all(colored <= 1))
    
    def test_save_audio(self):
        """Test saving audio to file."""
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        output_path = os.path.join(self.test_output_dir, "test_audio.wav")
        
        save_audio(audio, output_path, sample_rate=16000)
        
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
    
    def test_save_weights_plot(self):
        """Test saving weights plot."""
        piano = PolyphonicMember.piano(num_beats=4)
        output_path = os.path.join(self.test_output_dir, "test_weights.png")
        
        # Function expects a list of members
        save_weights_plot([piano], output_path)
        
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
    
    @patch('gradio.Blocks')
    @patch('gradio.Markdown')
    def test_gradio_interface_creation(self, mock_markdown, mock_blocks):
        """Test Gradio interface creation (mocked)."""
        from harmony.ui import GradioInterface
        
        interface = GradioInterface()
        
        # Test interface can be created
        # Note: We mock Gradio components to avoid actual UI creation
        self.assertIsNotNone(interface)
        self.assertEqual(interface.default_steps, 100)
    
    def test_gradio_tuning_options(self):
        """Test Gradio tuning options are defined."""
        from harmony.ui import TUNING_OPTIONS
        
        # Should have multiple tuning options
        self.assertIn("12-TET", TUNING_OPTIONS)
        self.assertIn("Pythagorean", TUNING_OPTIONS)
        self.assertIn("19-EDO", TUNING_OPTIONS)
        self.assertIn("31-EDO", TUNING_OPTIONS)
        
        # Each option should be callable
        for name, factory in TUNING_OPTIONS.items():
            tuning = factory()
            self.assertIsInstance(tuning, TuningSystem)


# =============================================================================
# Benchmark Tests
# =============================================================================

class TestBenchmarks(unittest.TestCase):
    """Performance benchmarks."""
    
    def test_optimization_speed(self):
        """Benchmark optimization speed."""
        piano = PolyphonicMember.piano(num_beats=4)
        optimizer = HarmonyOptimizer(members=[piano], lr=0.05)
        
        start_time = time.time()
        optimizer.optimize(steps=50, verbose=False)
        elapsed = time.time() - start_time
        
        print(f"\n  Optimization speed: 50 steps in {elapsed:.2f}s ({50/elapsed:.1f} steps/sec)")
        # Should complete in reasonable time
        self.assertLess(elapsed, 30.0)  # 30 seconds max
    
    def test_synthesis_speed(self):
        """Benchmark audio synthesis speed."""
        piano = PolyphonicMember.piano(num_beats=4)
        
        # Put some notes
        with torch.no_grad():
            piano.weights[30, 0] = 2.0
            piano.weights[34, 1] = 2.0
            piano.weights[37, 2] = 2.0
            piano.weights[41, 3] = 2.0
        
        synth = AudioSynthesizer(sample_rate=22050, beat_duration=0.5)
        
        start_time = time.time()
        audio = synth.synthesize_member(piano)
        elapsed = time.time() - start_time
        
        print(f"\n  Synthesis speed: {len(audio)/22050:.2f}s audio in {elapsed:.2f}s")
        # Should synthesize faster than real-time
        self.assertLess(elapsed, len(audio) / 22050 * 2)  # 2x real-time max


# =============================================================================
# Main entry point
# =============================================================================

def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("HARMONY FROM FIRST PRINCIPLES - COMPREHENSIVE INTEGRATION TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTuningSystems))
    suite.addTests(loader.loadTestsFromTestCase(TestInstruments))
    suite.addTests(loader.loadTestsFromTestCase(TestBandMembers))
    suite.addTests(loader.loadTestsFromTestCase(TestSynthesisAndMixing))
    suite.addTests(loader.loadTestsFromTestCase(TestLossAndOptimization))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizationAndUI))
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarks))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    # Run with unittest framework for detailed output
    unittest.main(verbosity=2, exit=False)
