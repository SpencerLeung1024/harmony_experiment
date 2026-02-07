# 🎵 Harmony From First Principles

A comprehensive musical optimization framework that generates harmonious music using physics-based dissonance minimization. Supports arbitrary tuning systems, multiple instrument types, and real-time optimization.

## 🌟 Features

### Tuning Systems
- **12-TET**: Standard Western tuning (A4=440Hz)
- **Pythagorean Tuning**: Pure perfect fifths (3:2 ratio)
- **Meantone Temperaments**: Quarter-comma and third-comma meantone
- **Equal Division of the Octave (EDO)**: 19-EDO, 24-EDO, 31-EDO, 41-EDO, 53-EDO
- **Non-Octave Scales**: Alpha scale (golden ratio), Beta scale (√2), Bohlen-Pierce (3:1)

### Instruments with ADSR Envelopes
- **Piano**: Rich harmonic content with quick attack
- **Guitar**: Plucked string timbre with natural decay
- **Bass**: Deep tones with strong fundamentals
- **Synth**: Synthetic timbres for electronic music
- **Drums**: Kick, snare, and hi-hat with specialized envelopes
- **Per-Harmonic ADSR**: Different envelope for each harmonic partial

### Band Members
- **PolyphonicMember**: Can play multiple notes simultaneously (e.g., piano)
- **MonophonicMember**: Only one note at a time (e.g., guitar, bass)
- **DrumMember**: Fixed pattern-based playback with standard rock patterns

### Optimization Features
- **Multi-Member Optimization**: Optimize multiple instruments together
- **Cross-Member Dissonance**: Minimize dissonance between different instruments
- **Temporal Dissonance**: Consider dissonance between adjacent beats
- **Density Control**: Target note density per member
- **Sparsity Penalty**: Encourage concentrated note choices
- **Range Constraints**: Keep notes within instrument range
- **Interval Jump Penalty**: Encourage smooth melodic lines
- **User Constraints**: Fix specific notes (like partial denoising)

### Audio Pipeline
- **Additive Synthesis**: Generate audio using sine wave harmonics
- **ADSR Envelopes**: Per-harmonic envelope shaping
- **Multi-Track Mixing**: Mix multiple instruments with gain control
- **Soft Limiting**: Prevent clipping while preserving dynamics

### Visualization
- **Weight Heatmaps**: See what notes each instrument plays
- **Piano Roll Views**: Traditional music notation visualization
- **Pitch-Class Coloring**: Color notes by their pitch class
- **Loss Curves**: Track optimization progress
- **Dissonance Matrices**: Visualize dissonance between notes
- **Spectrograms**: Frequency analysis of generated audio

### User Interfaces
- **Command-Line Demo**: `python demo.py` - Headless operation
- **Full Feature Demo**: `python run_full_demo.py` - Comprehensive showcase
- **Gradio Web UI**: `python app.py` - Interactive web interface

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd harmony_experiment

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from harmony import *

# Create a piano
piano = PolyphonicMember.piano(num_beats=4)

# Create optimizer
optimizer = HarmonyOptimizer(
    members=[piano],
    tuning=TwelveTET(),
    target_density=0.15
)

# Optimize
optimizer.optimize(num_steps=100)

# Synthesize audio
synth = AudioSynthesizer()
audio = synth.synthesize_member(piano)

# Save audio
save_audio(audio, "output.wav")
```

### Run Demos

```bash
# Quick CLI demo
python demo.py --members piano guitar --tuning 12-TET --steps 100

# Full feature showcase
python run_full_demo.py

# Interactive web interface
python app.py
```

## 📊 Examples

### Example 1: Piano Solo in 12-TET

```python
from harmony import *

piano = PolyphonicMember.piano(num_beats=4)
optimizer = HarmonyOptimizer(
    members=[piano],
    tuning=TwelveTET(),
    lr=0.03,
    target_density=0.15
)

loss_history = optimizer.optimize(num_steps=100)
```

### Example 2: Piano + Guitar in Pythagorean Tuning

```python
piano = PolyphonicMember.piano(num_beats=4)
guitar = MonophonicMember.guitar(num_beats=4)

optimizer = HarmonyOptimizer(
    members=[piano, guitar],
    tuning=PythagoreanTuning(),
    loss_weights={'cross': 1.0, 'within': 1.0}
)

optimizer.optimize(num_steps=100)
```

### Example 3: Full Band in 19-EDO

```python
piano = PolyphonicMember.piano(num_beats=4)
guitar = MonophonicMember.guitar(num_beats=4)
bass = MonophonicMember.bass(num_beats=4)
drums = DrumMember.standard_rock(num_beats=4)

optimizer = HarmonyOptimizer(
    members=[piano, guitar, bass, drums],
    tuning=EDOSystem(divisions=19)
)

optimizer.optimize(num_steps=100)
```

### Example 4: User Constraints (Fixed Notes)

```python
piano = PolyphonicMember.piano(num_beats=4)

# Fix C major chord on beat 0
constraints = ConstraintSet([
    UserConstraint("piano", 0, [39, 43, 46], fixed_value=1.0)  # C4, E4, G4
])

optimizer = HarmonyOptimizer(
    members=[piano],
    constraints=constraints
)

optimizer.optimize(num_steps=100)
```

## 🏗️ Architecture

```
harmony/
├── tuning.py          # Tuning systems (12-TET, Pythagorean, EDO, etc.)
├── instruments.py     # ADSR envelopes and instrument profiles
├── dissonance.py      # Dissonance calculation
├── band.py           # Band member types (Polyphonic, Monophonic, Drum)
├── losses.py         # Loss functions for optimization
├── constraints.py    # User constraints
├── optimizer.py      # Main optimization orchestration
├── synthesis.py      # Audio synthesis
├── mixer.py          # Audio mixing
├── visualization.py  # Plotting and visualization
└── ui.py            # Gradio web interface

app.py                # Entry point for web UI
demo.py               # CLI demo script
run_full_demo.py      # Comprehensive feature showcase
test_full_integration.py  # Complete test suite
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest test_full_integration.py -v

# Run specific test suite
python -m pytest test_full_integration.py::TestTuningSystems -v

# Run with coverage
python -m pytest test_full_integration.py --cov=harmony --cov-report=html
```

## 📚 Theory

### Dissonance Calculation

The dissonance between two pure tones is calculated using the psychoacoustic model:

```
d(f1, f2) = 65 * x * exp(-24 * x)
where x = |f2 - f1| / min(f1, f2)
```

This captures the phenomenon that two sine waves are:
- Consonant if almost exactly the same frequency
- Most dissonant when frequencies differ by about 50 cents (quarter tone)
- Less dissonant when far apart

### Optimization Process

1. **Initialize**: Random weights for each band member
2. **Calculate Dissonance**: Precompute dissonance matrices for efficiency
3. **Compute Loss**: Weighted sum of:
   - Within-member dissonance (simultaneous notes)
   - Temporal dissonance (adjacent beats)
   - Cross-member dissonance (between instruments)
   - Density penalty (target note density)
   - Sparsity penalty (concentrated weights)
   - Range penalty (instrument range)
   - Interval jump penalty (melodic smoothness)
4. **Optimize**: Adam optimizer with learning rate scheduling
5. **Synthesize**: Generate audio using additive synthesis

## 🔬 Scientific Background

This project is based on the scientific observation that:

> All sound can be expressed as sums of sine waves. Two sine waves are perceived as:
> - Consonant if almost exactly the same frequency
> - Dissonant if near each other, with a peak in dissonance at around 50 cents
> - Less dissonant if far away from each other

Surprisingly, just intervals don't matter for pure sine waves. Those patterns in music only show up once overtones from real instruments are involved.

## 🎛️ Configuration

### Loss Weights

Control the relative importance of different loss terms:

```python
loss_weights = {
    'within': 1.0,        # Within-member dissonance
    'temporal': 0.5,      # Temporal dissonance
    'cross': 1.0,         # Cross-member dissonance
    'density': 10.0,      # Density penalty
    'sparsity': 1.0,      # Sparsity penalty
    'range': 1.0,         # Range penalty
    'interval_jump': 0.5  # Interval jump penalty
}
```

### Optimization Parameters

```python
HarmonyOptimizer(
    members=members,
    lr=0.03,                    # Learning rate
    temporal_decay=0.3,         # Temporal dissonance weight
    target_density=0.15,        # Target note density
    enable_scheduler=True,      # Learning rate scheduling
    scheduler_step_size=50,     # Steps between LR decay
    scheduler_gamma=0.5         # LR decay factor
)
```

## 📝 Command-Line Options

### demo.py

```bash
python demo.py [options]
  --members {piano,guitar,bass,drums} [{...}]
                        Band members to include (default: piano)
  --tuning {12-tet,pythagorean,quarter-meantone,third-meantone,19-edo,24-edo,31-edo,41-edo,53-edo,alpha,beta,bohlen-pierce}
                        Tuning system (default: 12-tet)
  --steps N             Number of optimization steps (default: 100)
  --learning-rate LR    Learning rate (default: 0.02)
  --density D           Target note density 0-1 (default: 0.15)
  --beats N             Number of beats (default: 4)
  --output-dir DIR      Output directory (default: ./output/demo_TIMESTAMP)
```

### run_full_demo.py

```bash
python run_full_demo.py [options]
  --quick, -q           Quick mode with fewer optimization steps
  --no-tuning-comparison
                        Skip tuning system comparison
```

## 🤝 Contributing

This project demonstrates:
- Modern PyTorch optimization techniques
- Audio signal processing
- Music theory and tuning systems
- Scientific computing

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Credits

**Concept**: Harmony From First Principles

The fundamental insight that harmony can be derived from first principles of physics and psychoacoustics, rather than learned rules. This approach:
- Starts with sine waves as fundamental building blocks
- Uses measurable dissonance between frequencies
- Optimizes for consonance using gradient descent
- Discovers harmonic relationships emergently

All Future Plans from the original concept have been implemented:
- ✅ Alternate tuning systems (Pythagorean, Meantone, EDO, non-octave)
- ✅ ADSR instruments with per-harmonic envelopes
- ✅ Multiple band members (piano, guitar, bass, drums)
- ✅ Cross-member dissonance
- ✅ User constraints (fixed notes)
- ✅ Gradio web interface
- ✅ Comprehensive visualization

---

**Made with 🎵 and PyTorch**
