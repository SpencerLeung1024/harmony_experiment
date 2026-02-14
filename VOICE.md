# Voice Synthesis: From Organ to Ahh

This document explains how the `VoiceInstrument` creates vocal sounds, why early attempts sounded like organs or bowed strings, and what makes a synthesized sound be perceived as "voice-like."

## 1. The Voice Recognition Puzzle

Human listeners can instantly distinguish voices from instruments, even when both produce pitched sounds with harmonic spectra. Why?

### 1.1 What Makes Voice "Voice-like"

| Feature | Instruments | Voice | Perceptual Effect |
|---------|-------------|-------|-------------------|
| **Formants** | Fixed by physical structure | Shift with pitch | "Whoop" quality at pitch extremes |
| **Vibrato** | Usually 0 or periodic | ~5-7 Hz, ±3-6%, irregular | "Living" quality |
| **Noise/Breath** | Minimal or absent | Present at attacks | Consonants, note onsets |
| **Inharmonicity** | Usually perfect harmonics | Slight stretch at high freqs | Natural "roughness" |
| **Envelope** | Fast attack or gradual bow | Breath-like, variable | Expressiveness |

## 2. Formants: The Spectral Fingerprint

### 2.1 What Are Formants?

Formants are **resonant frequency bands** in the vocal tract. Unlike harmonics (which are integer multiples of the fundamental), formants are **fixed frequencies** that don't change when pitch changes.

Think of the vocal tract as a tube with adjustable diameter:
- **Tongue position** creates constrictions
- **Lip rounding** lengthens/shortens the tube
- **Jaw opening** changes the overall shape

These constrictions create resonances at specific frequencies—the formants.

### 2.2 The Vowel Triangle

Different vowels are created by positioning the first two formants (F1 and F2):

```
F2 (Hz)
  ↑
2500 │    /i/ (ee)        /u/ (oo)
     │      ↗                ↖
2000 │        ↗            ↖
     │          ↗        ↖
1500 │            ↗    ↖
     │              ↗↖
1000 │              /a/ (ah)
     │
 700 │    /ʌ/ (uh)
     │
 300 │/ɑ/ (aw)
     └──────────────────────────→ F1 (Hz)
       300    700    1000   1500
```

**Key insight:** The *distance* between F1 and F2 determines vowel quality:
- **Wide F1-F2 spacing** = "bright" vowels (ee, eh)
- **Close F1-F2 spacing** = "dark" vowels (oo, oh)

### 2.3 Formant Frequencies for Common Vowels

| Vowel | F1 (Hz) | F2 (Hz) | F3 (Hz) | F4 (Hz) | Description |
|-------|---------|---------|---------|---------|-------------|
| /i/ (ee) | 280 | 2200 | 3000 | 3600 | "see" |
| /ɪ/ (ih) | 350 | 2100 | 2900 | 3500 | "sit" |
| /e/ (ay) | 450 | 2000 | 2800 | 3400 | "say" |
| /æ/ (ah) | 700 | 1700 | 2600 | 3200 | "sat" |
| /ɑ/ (aw) | 750 | 1150 | 2650 | 3500 | "father" |
| /ʌ/ (uh) | 700 | 1200 | 2650 | 3500 | "cup" |
| /o/ (oh) | 500 | 900 | 2500 | 3300 | "boat" |
| /u/ (oo) | 300 | 700 | 2300 | 3200 | "boot" |

**Note:** These are approximate values for an adult male voice. Female voices are typically ~15% higher. Children's voices can be 25-30% higher.

### 2.4 Why Formants Don't Scale with Pitch

If you sing "ah" at A2 (110 Hz) and then at A4 (440 Hz), the formant frequencies stay roughly the same. The harmonics shift (110→220→330... becomes 440→880→1320...), but the **spectral envelope**—which frequencies are boosted—remains constant.

This is why:
- We can understand speech at any pitch
- A child and adult singing the same vowel have the same "ah-ness"
- Chipmunk voices still sound like the original speaker

## 3. The "Whoop" Effect: Formant Shifting

### 3.1 Real Voices Shift Formants

While formants are *approximately* fixed, they do shift slightly with pitch. Research shows:

- **Sopranos** shift F1 up by 100-200 Hz at high notes
- **Basses** shift F1 down at low notes
- **Typical shift rate:** 10-20% per octave

This is called the **whoop effect** because it sounds like a "whoop" when exaggerated.

### 3.2 Why Formants Shift

When singing high notes:
1. The larynx rises slightly
2. The vocal tract shortens
3. All resonant frequencies increase

When singing low notes:
1. The larynx lowers
2. The vocal tract lengthens
3. All resonant frequencies decrease

### 3.3 Implementation in VoiceInstrument

```python
# Base formants at reference pitch (A3 = 220 Hz)
base_formants = [
    (750, 1.0, 80),    # F1 at reference
    (1150, 0.7, 110),  # F2 at reference
    (2650, 0.35, 160), # F3 at reference
    (3500, 0.18, 220), # F4 at reference
]

# Shift factor: how much formants move per octave
shift_rate = 0.15  # 15% per octave

# For a given pitch:
octaves = np.log2(freq / 220.0)  # Octaves from A3
shift_factor = 1.0 + shift_rate * octaves

# Shifted formant frequency:
F1_shifted = 750 * shift_factor
```

At A2 (110 Hz, one octave below): F1 ≈ 640 Hz (down 15%)
At A4 (440 Hz, one octave above): F1 ≈ 863 Hz (up 15%)

## 4. The Source-Filter Model

Voice synthesis uses a **source-filter model**:

```
[Glottal Source] → [Vocal Tract Filter] → [Output]
     ↓                    ↓
  Sawtooth-like       Formant peaks
  spectrum            in frequency
```

### 4.1 The Glottal Source

Real vocal folds don't produce perfect sawtooth waves. They produce:
- **A pulse train** with gradual opening, sudden closing
- **Amplitude envelope:** -12 dB/octave rolloff (steeper than sawtooth's -6 dB)
- **Subharmonics:** Sometimes period-doubling occurs, creating 0.5×, 1.5× frequencies

Our implementation approximates this with a sawtooth-like source (1/n amplitudes) with additional rolloff above 4 kHz.

### 4.2 The Formant Filter

The vocal tract acts as a series of resonators. We model this by:

1. Generating harmonics at frequencies `n × fundamental`
2. For each harmonic, calculating its distance from each formant
3. Applying a **combined Gaussian-Lorentzian weighting**:

```python
for formant_freq, formant_amp, bandwidth in formants:
    distance = abs(harmonic_freq - formant_freq)
    
    # Gaussian component (smooth peak)
    gauss = exp(-0.5 * (distance / bandwidth)²)
    
    # Lorentzian component (broader tails)
    lorentz = bandwidth² / (distance² + bandwidth²)
    
    # Combined (70% Gaussian, 30% Lorentzian)
    weight = formant_amp * (0.7 * gauss + 0.3 * lorentz)
```

**Why combine them?**
- Pure Gaussian: Too smooth, sounds artificial
- Pure Lorentzian: Too broad, sounds muffled
- Combined: Matches measured vocal tract resonances

## 5. Inharmonicity: The Natural Imperfection

### 5.1 What Is Inharmonicity?

In a perfect oscillator, harmonics are at exactly 2×, 3×, 4× the fundamental. Real vocal folds have **stiffness**, making higher harmonics slightly sharp:

```
f_n = n × f₀ × √(1 + B × n²)
```

Where `B` is the inharmonicity coefficient (~0.001 for voices).

### 5.2 Effect on Perception

- **Perfect harmonics:** Sound "electronic," "sterile"
- **Slightly stretched:** Sound "natural," "organic"
- **Too stretched:** Sound "detuned," "chorused"

At B = 0.001:
- 2nd harmonic: +0.05% sharp (barely perceptible)
- 10th harmonic: +1.0% sharp (slight "bite")
- 20th harmonic: +4.0% sharp (noticeable "air")

### 5.3 Implementation

```python
def harmonic_frequency(n, fundamental, B=0.001):
    stretch = sqrt(1.0 + B * n * n)
    return fundamental * n * stretch
```

## 6. Vibrato: The "Living" Quality

### 6.1 What Is Vibrato?

Vibrato is **periodic pitch modulation**:
- **Rate:** 5-7 Hz (average ~5.5 Hz)
- **Depth:** ±3-6% of pitch (about ±50-100 cents)
- **Waveform:** Approximately sinusoidal but irregular

### 6.2 Why Vibrato Matters

Without vibrato:
- Voices sound "dead," "synthetic," "robotic"
- Instruments sound "clinical," "sampled"

With vibrato:
- Adds "warmth" and "expression"
- Masks intonation imperfections
- Creates spectral "smearing" that softens harshness

### 6.3 Implementation

```python
# Vibrato parameters
vibrato_rate = 5.5    # Hz
vibrato_depth = 0.03  # ±3%

# Modulation over time
vibrato = 1.0 + vibrato_depth * sin(2π × vibrato_rate × t)

# Apply to each harmonic (higher harmonics get more vibrato)
for n in range(1, num_harmonics + 1):
    # Higher harmonics have wider absolute excursion
    harmonic_vibrato = 1.0 + (vibrato_depth * (1.0 + 0.1 * n)) * sin(phase)
    freq = base_freq * n * harmonic_vibrato
```

**Note:** Higher harmonics get more vibrato excursion in absolute Hz (but same percentage), which matches real voices.

## 7. Breath Noise: The Attack Character

### 7.1 Aspiration in Voice

When vocal folds first engage, they don't close perfectly. This creates:
- **Turbulence noise** (broadband 1-5 kHz)
- **Gradual amplitude rise** (not instant)
- **Partial voicing** (mixed periodic + noise)

### 7.2 Implementation

```python
# Generate white noise
noise = random.randn(samples) * breath_amount

# Breath envelope: strongest at attack, decays over ~150ms
breath_envelope = exp(-t / 0.15)

# High-pass characteristic (aspiration is trebly)
# Simple approximation: full spectrum noise modulated by envelope
breath = noise * breath_envelope

# Add to harmonic signal
output = harmonic_signal + breath
```

## 8. Why Early Attempts Sounded Like Organs

### 8.1 The Organ Problem

The original `VoiceInstrument` had:
1. **Fixed formants** → No pitch-dependent character change
2. **No vibrato** → Perfectly static pitch
3. **Perfect harmonics** → Too "clean"
4. **Gaussian-only formants** → Too smooth, no natural resonance tails
5. **No breath noise** → Instant, pure tone attacks

Pipe organs also have:
1. Fixed resonators (the pipes)
2. No vibrato (usually)
3. Nearly perfect harmonics
4. Smooth spectral envelopes
5. Clean attacks (air valves opening)

The similarity was structural, not coincidental.

### 8.2 Bowed String Similarity

Bowed strings share with the original voice implementation:
- Fixed body resonances (like fixed formants)
- Continuous excitation (like sustained voice)
- Slight inharmonicity (from string stiffness)

The main difference: bows create **sawtooth-like** waveforms with all harmonics, similar to our original source.

## 9. Parameter Tuning Guide

### 9.1 Choir "Aah" (`voice_aah`)

```python
VoiceInstrument(
    formants=[
        (750, 1.00, 85),    # F1: bright, open
        (1150, 0.70, 110),  # F2: well-separated from F1
        (2650, 0.35, 160),  # F3: high
        (3500, 0.18, 220),  # F4: adds "sheen"
    ],
    vibrato_rate=5.5,
    vibrato_depth=0.03,      # Moderate vibrato
    breath_amount=0.15,      # Noticeable breath
    inharmonicity=0.001,     # Slight stretch
    formant_shift_rate=0.15  # Natural whoop
)
```

### 9.2 Choir "Ooh" (`voice_ooh`)

```python
VoiceInstrument(
    formants=[
        (300, 1.00, 60),     # F1: very low (dark)
        (700, 0.55, 90),     # F2: close to F1
        (2300, 0.30, 140),   # F3
        (3200, 0.15, 200),   # F4
    ],
    vibrato_rate=5.2,        # Slightly slower
    vibrato_depth=0.025,     # Gentler vibrato
    breath_amount=0.12,      # Less breath (rounded vowel)
    inharmonicity=0.0008,
    formant_shift_rate=0.12  # Less shift (closed mouth)
)
```

### 9.3 Solo Soprano (`voice_eeh`)

```python
VoiceInstrument(
    formants=[
        (280, 1.00, 55),     # F1: low ("ee" is bright but F1 is low)
        (2200, 0.80, 130),   # F2: very high (the "brightness")
        (3000, 0.40, 170),   # F3
        (3600, 0.20, 230),   # F4
    ],
    vibrato_rate=5.8,        # Faster (expressive solo)
    vibrato_depth=0.035,     # Wider (soloist freedom)
    breath_amount=0.10,      # Less breath (trained voice)
    inharmonicity=0.0012,    # More stretch (higher pitch)
    formant_shift_rate=0.18  # More shift (trained soprano technique)
)
```

## 10. Limitations and Future Directions

### 10.1 What's Missing

1. **Formant transitions:** Real voices interpolate formants when changing vowels
2. **Pitch scooping:** Voices often "scoop" up to target pitch at note onset
3. **Vibrato irregularity:** Real vibrato varies in rate and depth
4. **Multiple voices:** Choir = 20+ individual voices with slight variations
5. **Consonants:** /p/, /t/, /k/, /s/, etc. require different models entirely

### 10.2 Computational Limits

The current implementation is already at the edge of what's practical:
- 24 harmonics × 4 formants = 96 filter calculations per sample
- Vibrato requires phase integration (FM synthesis)
- Formant shifting adds overhead

More realism would require:
- **Convolution:** Pre-compute impulse responses
- **Machine learning:** Neural vocoders (WaveNet, HiFi-GAN)
- **Sampling:** Record real voices, pitch-shift

### 10.3 The Uncanny Valley of Voice

Voice synthesis has an **uncanny valley**:
- **Totally synthetic** (chipmunk, robot): Acceptable
- **Nearly realistic** (slightly off): Creepy, annoying
- **Fully realistic** (actual recordings): Ideal

Current synthesis sits in the "nearly realistic" zone for solo voices, but works well for choir textures where individual imperfections blend together.

## 11. Summary

To make a synthesizer sound "voice-like":

1. **Use formants** - Fixed resonances define vowel quality
2. **Shift formants with pitch** - 10-20% per octave for naturalness
3. **Add vibrato** - 5.5 Hz, ±3%, irregular
4. **Include breath noise** - Especially at note attacks
5. **Apply inharmonicity** - Slight stretch (B ≈ 0.001)
6. **Use complex formant shapes** - Gaussian + Lorentzian, not pure Gaussian
7. **Add a 4th formant** - Above 3 kHz for "air" and "sheen"

The `VoiceInstrument` implements all of these, transforming the organ-like original into something recognizably vocal.

---

*"The human voice is the most beautiful instrument of all, but also the most difficult to play."* — Not Richard Strauss, but someone should have said it.
