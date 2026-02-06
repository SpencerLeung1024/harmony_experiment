# Implementation Roadmap

## Phase 1: Foundation (Tuning + Instruments)
**Dependencies**: None (can be developed in parallel)

### 1.1 Tuning Systems Module
- [ ] `harmony/tuning/base.py`: `TuningSystem` ABC
- [ ] `harmony/tuning/twelve_tet.py`: `TwelveTET` class
- [ ] `harmony/tuning/pythagorean.py`: `PythagoreanTuning` class
- [ ] `harmony/tuning/meantone.py`: `MeantoneTuning` class
- [ ] `harmony/tuning/edo.py`: `EDOSystem` class
- [ ] `harmony/tuning/non_octave.py`: `NonOctaveSystem` class

**Key Tests**:
```python
# 12-TET: A4 = 440Hz
tet = TwelveTET()
assert abs(tet.get_frequency(69) - 440.0) < 0.01

# Pythagorean: Perfect fifths are 3/2 ratio
pyt = PythagoreanTuning()
f_C = pyt.get_frequency(60)  # Middle C
f_G = pyt.get_frequency(67)  # G above
assert abs((f_G / f_C) - 1.5) < 0.001
```

### 1.2 Instruments Module
- [ ] `harmony/instruments/base.py`: `Instrument` ABC, `Harmonic`, `ADSR`
- [ ] `harmony/instruments/synth.py`: `Synthesizer` (default 6 harmonics)
- [ ] `harmony/instruments/guitar.py`: `Guitar` (with inharmonicity)
- [ ] `harmony/instruments/bass.py`: `Bass` (fewer harmonics)
- [ ] `harmony/instruments/drums.py`: `Drum` (noise-based)

**Key Tests**:
```python
# Synth has expected harmonics
synth = Synthesizer()
harmonics = synth.get_harmonics(440.0)
assert len(harmonics) == 6
assert harmonics[0].ratio == 1.0  # Fundamental
assert harmonics[1].ratio == 2.0  # Octave

# ADSR envelope generation
adsr = ADSR(attack=0.01, decay=0.1, sustain_level=0.7, release=0.3)
env = adsr.get_envelope(duration=1.0, sample_rate=22050)
assert env.shape[0] == 22050
assert env[0] < env[int(0.01 * 22050)]  # Attack increases
```

---

## Phase 2: Core Engine (Dissonance + Synthesis)
**Dependencies**: Phase 1

### 2.1 Dissonance Calculator
- [ ] `harmony/dissonance/calculator.py`: `DissonanceCalculator`
  - [ ] `calculate_matrix()` - within-member dissonance
  - [ ] `calculate_cross_matrix()` - cross-member dissonance
  - [ ] Caching mechanism for repeated calculations

**Key Tests**:
```python
# Same note has zero dissonance with itself
calc = DissonanceCalculator()
tet = TwelveTET()
synth = Synthesizer()
D = calc.calculate_matrix(tet, synth, (60, 72))  # C4 to B4
assert D[0, 0] == 0.0  # Same note, no self-dissonance

# Octave has low dissonance (same pitch class, 12 semitones)
C4_idx = 0  # Key 60
C5_idx = 12  # Key 72
assert D[C4_idx, C5_idx] < D[C4_idx, C5_idx - 1]  # Less than minor 7th
```

### 2.2 Audio Synthesis
- [ ] `harmony/synthesis/envelopes.py`: ADSR envelope application
- [ ] `harmony/synthesis/engine.py`: `AudioSynthesizer`
  - [ ] Additive synthesis with harmonics
  - [ ] Per-note ADSR envelope
  - [ ] Beat timing alignment

**Key Tests**:
```python
# Synthesized audio has correct duration
synth_engine = AudioSynthesizer(sample_rate=22050)
weights = torch.zeros((12, 4))  # 12 keys, 4 beats
weights[0, 0] = 1.0  # C4 on beat 0

audio = synth_engine.synthesize(
    member=member, weights=weights,
    beat_duration=0.5, total_duration=2.0
)
assert audio.shape[0] == 2.0 * 22050
```

### 2.3 Audio Mixer
- [ ] `harmony/band/mixer.py`: `AudioMixer`
  - [ ] Gain control per member
  - [ ] Stereo panning (optional)
  - [ ] Peak normalization

---

## Phase 3: Band Members
**Dependencies**: Phase 1, Phase 2

### 3.1 Base Member Classes
- [ ] `harmony/band/base.py`: `BandMember` ABC
- [ ] `harmony/band/piano.py`: `PolyphonicMember` (Piano)
- [ ] `harmony/band/guitar.py`: `MonophonicMember` (Guitar)
- [ ] `harmony/band/bass.py`: `MonophonicMember` (Bass)
- [ ] `harmony/band/drums.py`: `DrumMember` (Drums)

**Key Tests**:
```python
# Piano allows multiple notes per beat
piano = PolyphonicMember("piano", Synthesizer(), TwelveTET(), num_beats=8)
raw_weights = torch.rand((88, 8))
prepared = piano.prepare_weights(raw_weights)
# Multiple notes can be active
assert (prepared[:, 0] > 0.1).sum() > 1

# Guitar selects single note per beat
guitar = MonophonicMember("guitar", Guitar(), TwelveTET(), num_beats=8)
raw_weights = torch.rand((24, 8))
prepared = guitar.prepare_weights(raw_weights)
# Only one note active per beat (approximately)
for b in range(8):
    active = (prepared[:, b] > 0.5).sum()
    assert active == 1 or active == 0

# Drums have fixed pattern
drums = DrumMember("drums", pattern="basic_rock", num_beats=8)
fixed = drums.get_fixed_pattern()
# Kick on beats 0, 2, 4, 6; Snare on 1, 3, 5, 7
assert fixed[0, 0] > 0  # Kick beat 0
assert fixed[0, 1] == 0  # No kick beat 1
assert fixed[1, 1] > 0  # Snare beat 1
```

---

## Phase 4: Optimization Engine
**Dependencies**: Phase 1, Phase 2, Phase 3

### 4.1 Constraints
- [ ] `harmony/optimization/constraints.py`: `UserConstraint`
  - [ ] Fixed weight masking
  - [ ] Beat range specification

### 4.2 Loss Functions
- [ ] `harmony/optimization/losses.py`: `LossFunction`
  - [ ] Within-beat dissonance (vectorized)
  - [ ] Temporal dissonance (adjacent beats)
  - [ ] Cross-member dissonance
  - [ ] Density penalty
  - [ ] Range penalty
  - [ ] Interval jump penalty

**Key Tests**:
```python
# Within-beat loss is zero for single note
weights = torch.zeros((12, 4))
weights[0, 0] = 1.0
loss_fn = LossFunction()
loss = loss_fn.calculate(members=[member], weights={"test": weights}, ...)
assert loss.item() == 0.0  # No dissonance with single note

# Two notes a semitone apart have high loss
weights[1, 0] = 1.0  # Add minor second
loss_with_semitone = loss_fn.calculate(...)
assert loss_with_semitone > loss
```

### 4.3 Optimizer
- [ ] `harmony/optimization/optimizer.py`: `HarmonyOptimizer`
  - [ ] Weight initialization
  - [ ] Dissonance matrix precomputation
  - [ ] Training loop with progress reporting
  - [ ] Learning rate scheduling

**Key Tests**:
```python
# Optimizer reduces loss over time
optimizer = HarmonyOptimizer(tuning, members, steps=50)
initial_loss = optimizer.step()
for _ in range(49):
    final_loss = optimizer.step()
assert final_loss < initial_loss
```

---

## Phase 5: Visualization & UI
**Dependencies**: Phase 1-4

### 5.1 Visualizations
- [ ] `harmony/ui/visualizations.py`:
  - [ ] Weight matrix heatmap with pitch class coloring
  - [ ] Spectrogram with pitch class coloring
  - [ ] Loss curve plotting
  - [ ] Chord progression display

### 5.2 Gradio Interface
- [ ] `harmony/ui/gradio_app.py`: `GradioInterface`
  - [ ] Tuning system selector
  - [ ] Band member configuration (add/remove)
  - [ ] Parameter sliders (LR, steps, loss weights)
  - [ ] Piano roll for user constraints
  - [ ] Optimization progress bar
  - [ ] Audio player and download
  - [ ] Visualization display

**Key UI Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  Harmony From First Principles                              │
├─────────────────────────────────────────────────────────────┤
│  [Setup] [Piano Roll] [Optimize] [Results]                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Tuning System: [12-TET ▼]                                  │
│                                                             │
│  Band Members:                                              │
│  ☑️ Piano (88 keys, 8 beats)                               │
│  ☑️ Guitar (24 keys, 16 beats)                             │
│  ☐ Bass                                                   │
│  ☑️ Drums (fixed pattern)                                  │
│                                                             │
│  Optimization Steps: [200 ────────────────]                 │
│  Learning Rate: [0.02 ────────────────]                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 6: Integration & Testing
**Dependencies**: All previous phases

### 6.1 Integration Examples
- [ ] `examples/basic_optimization.py`: Single piano, 12-TET
- [ ] `examples/alternate_tuning.py`: Compare 12-TET vs Pythagorean
- [ ] `examples/full_band.py`: Piano + Guitar + Bass + Drums
- [ ] `examples/user_constraints.py`: Fixed melody with optimized harmony

### 6.2 Unit Tests
- [ ] `tests/test_tuning.py`: All tuning systems
- [ ] `tests/test_instruments.py`: All instruments
- [ ] `tests/test_dissonance.py`: Dissonance calculation correctness
- [ ] `tests/test_synthesis.py`: Audio generation
- [ ] `tests/test_optimization.py`: Loss reduction

### 6.3 End-to-End Tests
- [ ] Optimization produces valid audio
- [ ] Loss consistently decreases
- [ ] Different tunings produce different results
- [ ] Multi-member setup works correctly

---

## Development Order Summary

```
Week 1: Foundation
├── Day 1-2: Tuning systems
├── Day 3-4: Instruments + ADSR
└── Day 5: Tests + Integration

Week 2: Core Engine
├── Day 1-2: Dissonance calculator
├── Day 3-4: Audio synthesis + mixer
└── Day 5: Tests + Integration

Week 3: Band Members + Optimization
├── Day 1-2: Band member classes
├── Day 3-4: Loss functions + optimizer
└── Day 5: Tests + Integration

Week 4: UI + Polish
├── Day 1-2: Visualizations
├── Day 3-4: Gradio interface
└── Day 5: Examples + Documentation
```

---

## Success Criteria

1. **Correctness**: Dissonance calculations match original `main_optimized.py` for 12-TET
2. **Performance**: Optimization runs at similar speed to original
3. **Flexibility**: Can switch tuning systems without code changes
4. **Extensibility**: New band member can be added in <50 lines
5. **Usability**: Gradio interface allows full control without editing code
