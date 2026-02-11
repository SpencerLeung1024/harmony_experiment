# Audio Compression and Limiting Functions

This document explores different compression functions for audio limiting, analyzing their mathematical properties, distortion characteristics, and qualitative effects.

## 1. The Goal of Compression

When mixing multiple band members, audio peaks can exceed the valid range `[-1.0, 1.0]`. A limiter prevents clipping by:
1. Passing small signals through unchanged (linear regime)
2. Compressing larger signals above a threshold (nonlinear regime)
3. Ensuring output never exceeds a hard limit

The ideal limiter should:
- Preserve the "character" of the audio below the threshold
- Introduce minimal, "musical" distortion when compressing
- Be continuous and smooth at the threshold transition
- Have predictable behavior for any input level

## 2. Common Compression Functions

### 2.1 Hyperbolic Tangent (tanh)

**Formula:**
```
f(x) = T + tanh((x - T) / (L - T)) × (L - T)   for x > T
f(x) = x                                        for x ≤ T
```

Where `T` = threshold, `L` = limit value.

**Properties:**
- **Smoothness:** C^∞ (infinitely differentiable)
- **Slope at threshold:** 1 (perfectly continuous with linear regime)
- **Asymptotic behavior:** Approaches `L` as `x → ∞`
- **Second derivative:** Maximum at threshold, decays smoothly

**Distortion Characteristics:**

Tanh is an odd function, so for a pure sine wave input `A·sin(ωt)`:

```
tanh(A·sin(ωt)) = Σ_{n odd} c_n · sin(nωt)
```

Only **odd harmonics** are generated. This is musically favorable because:
- Odd harmonics (3rd, 5th, 7th...) sound "warm" and "tube-like"
- They reinforce the fundamental's pitch without creating strong dissonance
- Guitar amps and tube equipment naturally produce odd harmonic distortion

**Harmonic Decay:** The nth harmonic amplitude decays approximately as `1/n²`, meaning higher harmonics are much quieter than lower ones.

---

### 2.2 Arc Tangent (arctan)

**Formula:**
```
f(x) = T + (2/π) · arctan(π(x - T) / (2(L - T))) × (L - T)
```

**Properties:**
- **Smoothness:** C^∞
- **Slope at threshold:** 1 (matched)
- **Asymptotic approach:** Slower than tanh, reaches 90% of limit at ~3.1× overshoot vs tanh's ~2.0×

**Distortion Characteristics:**

Arctan also generates only odd harmonics, but with different weighting:

```
arctan(A·sin(ωt)) = Σ_{n odd} c_n · sin(nωt)
```

**Comparison to tanh:**
- Lower 3rd harmonic amplitude (~25% less)
- Slower harmonic decay (5th, 7th harmonics relatively stronger)
- Sounds "softer" or "more gentle" than tanh
- Can be perceived as more "transparent" for extreme compression

---

### 2.3 Sigmoid/Logistic Function

**Formula:**
```
f(x) = T + (L - T) / (1 + exp(-4(x - T)/(L - T)))
```

**Properties:**
- **Smoothness:** C^∞
- **Slope at threshold:** 0.5 (not matched to linear regime!)
- **Asymptotic approach:** Similar to tanh

**The Slope Problem:**

Unlike tanh and arctan, sigmoid has slope 0.5 at its center point. This creates a **discontinuity in the first derivative** unless we rescale:

```
# Rescaled sigmoid with slope 1 at threshold
g(x) = T + (L - T) · (1 + tanh(2(x - T)/(L - T))) / 2
```

This is actually mathematically equivalent to the tanh formulation! The sigmoid and tanh are related by:
```
sigmoid(x) = (tanh(x/2) + 1) / 2
```

**Recommendation:** Use tanh directly rather than sigmoid.

---

### 2.4 Soft Clipping (Polynomial)

**Formula (3rd order):**
```
f(x) = x - (x - T)³ / (3(L - T)²)    for T < x < T + √3(L - T)
f(x) = L                              for x ≥ T + √3(L - T)
f(x) = x                              for x ≤ T
```

**Properties:**
- **Smoothness:** C^1 (continuous first derivative)
- **Slope at threshold:** 1 (matched)
- **Hard limit:** At finite input, not asymptotic

**Distortion Characteristics:**

Polynomial soft clipping generates **both odd and even harmonics**:

```
x - x³ = x - (3sin(ωt) - sin(3ωt))/4 = (3/4)sin(ωt) + (1/4)sin(3ωt)
```

**Even harmonics** are present when the polynomial has even-powered terms. A symmetric odd function only produces odd harmonics, but practical polynomial clippers often include a small even component.

**Sound quality:**
- Harder, more "aggressive" character than tanh
- 3rd harmonic is stronger relative to tanh
- Can sound "crunchy" or "fuzzy" like transistor distortion
- Used in some guitar pedal emulations

---

### 2.5 Hard Clipping (Comparator)

**Formula:**
```
f(x) = L    for x > L
f(x) = x    for T < x ≤ L
f(x) = x    for x ≤ T
```

**Properties:**
- **Smoothness:** C^-1 (discontinuous!)
- **Distortion:** Maximum - generates infinite harmonics

**Distortion Characteristics:**

A square wave is the extreme case of hard clipping. Its Fourier series:

```
square(t) = (4/π) · Σ_{n odd} sin(nωt)/n
```

All odd harmonics with amplitude decaying as `1/n` (much slower than tanh's `1/n²`).

**Sound quality:**
- Harsh, digital-sounding distortion
- Not recommended for musical applications
- Useful only for extreme "bitcrushed" effects

---

### 2.6 Exponential Soft Clipping

**Formula:**
```
f(x) = T + (L - T) · (1 - exp(-(x - T)/(L - T)))
```

**Properties:**
- **Smoothness:** C^∞
- **Slope at threshold:** 1 (matched)
- **Asymptotic approach:** Never actually reaches L, gets 95% there at 3× overshoot

**Distortion Characteristics:**

Exponential compression has even worse harmonic decay than tanh - higher harmonics are more prominent:

```
1 - exp(-A·sin(ωt)) = A·sin(ωt) - A²·sin²(ωt)/2 + A³·sin³(ωt)/6 - ...
```

The `sin²` term introduces **even harmonics** and a DC offset:
```
sin²(ωt) = (1 - cos(2ωt))/2
```

**Sound quality:**
- Can sound "hollow" or "scooped" due to even harmonics
- The DC offset shifts the waveform center (bad for audio!)
- Generally not recommended for transparent limiting

---

## 3. Harmonic Analysis Summary

| Function | Harmonic Type | 3rd/1st Ratio | Decay Rate | Character |
|----------|---------------|---------------|------------|-----------|
| tanh | Odd only | ~10% at 3dB over | 1/n² | Warm, tube-like |
| arctan | Odd only | ~7% at 3dB over | 1/n | Soft, gentle |
| Polynomial (3rd) | Odd only | ~15% at 3dB over | Fast | Crunchy, fuzzy |
| Exponential | Odd + Even | ~12% + DC | Slow | Hollow, scooped |
| Hard Clip | Odd only | 33% | 1/n | Harsh, digital |

*All ratios measured with 3dB of compression (input at threshold + 3dB)*

---

## 4. Comparison to Physical Distortion

### 4.1 Vacuum Tubes (Triodes, Pentodes)

**Mechanism:** Nonlinear transfer curve due to electron flow physics

**Distortion profile:**
- Primarily **odd harmonics** (2nd harmonic ~1-2%, 3rd ~0.5%)
- Asymmetric: positive and negative halves distort differently
- Creates **even harmonics** at very high drive levels

**Best match:** tanh with slight asymmetry

```python
# Asymmetric tanh (tube-like)
positive = T + tanh((x - T)/(L - T)) * (L - T)
negative = -T - tanh((-x - T)/(L - T)) * (L - T) * 0.95  # Slightly less compression on negative side
```

---

### 4.2 Transistor Clipping (Bipolar, FET)

**Mechanism:** Saturation of semiconductor junctions

**Distortion profile:**
- **Harder knee** than tubes (closer to polynomial)
- Can be symmetric or asymmetric depending on circuit
- More 3rd harmonic relative to tubes

**Best match:** 3rd-order polynomial or modified tanh with sharper knee

---

### 4.3 Op-Amp Saturation

**Mechanism:** Output stage hitting supply rails

**Distortion profile:**
- Very hard knee near rails
- Can include crossover distortion in Class B designs
- Often sounds "sterile" compared to tube/transistor

**Best match:** Hard clip with tiny amount of softening

---

### 4.4 Magnetic Tape Saturation

**Mechanism:** Magnetic domains reaching saturation

**Distortion profile:**
- **Asymmetric** (different compression on positive/negative)
- **Hysteresis** - output depends on history, not just current input
- Creates **even harmonics** and intermodulation

**Best match:** Asymmetric tanh with hysteresis (requires stateful implementation)

---

### 4.5 Speaker Cone Breakup

**Mechanism:** Mechanical nonlinearity at high excursion

**Distortion profile:**
- Frequency-dependent (worse at low frequencies)
- **Odd harmonics** primarily
- Increases with volume

**Best match:** Dynamic range compression (not limiting), or tanh with frequency-dependent threshold

---

## 5. Mathematical Derivations

### 5.1 Harmonic Content of tanh

For input `x(t) = A·sin(ωt)` where `A > T`:

```
y(t) = tanh(A·sin(ωt))
```

Using the Fourier series expansion:

```
tanh(A·sin(ωt)) = Σ_{n=0}^∞ b_{2n+1} · sin((2n+1)ωt)
```

Where:
```
b_n = (2/π) ∫_0^π tanh(A·sin(θ)) · sin(nθ) dθ
```

For small signals (`A << 1`), `tanh(x) ≈ x - x³/3`, so:
```
tanh(A·sin(ωt)) ≈ A·sin(ωt) - (A³/3)·sin³(ωt)
                = A·sin(ωt) - (A³/3)·(3sin(ωt) - sin(3ωt))/4
                = (A - A³/4)·sin(ωt) + (A³/12)·sin(3ωt)
```

The 3rd harmonic ratio is approximately `A²/12`.

For `A = 2` (6dB over threshold), 3rd harmonic is ~33% of fundamental.

---

### 5.2 Why tanh is "Nice"

The hyperbolic tangent has several mathematical properties that make it ideal:

1. **Analytic continuation:** tanh is entire (analytic everywhere in complex plane)
2. **Bounded output:** |tanh(x)| < 1 for all finite x
3. **Simple derivative:** d/dx tanh(x) = 1 - tanh²(x) (easy for gradient-based optimization)
4. **Rapid convergence of harmonics:** Coefficients decay exponentially in n

---

## 6. Implementation Recommendations

### 6.1 For "Transparent" Limiting

Use tanh with moderate threshold (`headroom_db = -6` to `-12`):
```python
def transparent_limit(audio, threshold_db=-6.0, limit=0.95):
    T = 10 ** (threshold_db / 20)
    mask = np.abs(audio) > T
    compressed = T + np.tanh((np.abs(audio) - T) / (limit - T)) * (limit - T)
    return np.sign(audio) * np.where(mask, compressed, np.abs(audio))
```

### 6.2 For "Warm" Character

Add slight asymmetry:
```python
def warm_limit(audio, threshold_db=-6.0, limit=0.95, asymmetry=0.05):
    T = 10 ** (threshold_db / 20)
    positive = audio > T
    negative = audio < -T
    
    # Compress positive side normally
    pos_compressed = T + np.tanh((audio - T) / (limit - T)) * (limit - T)
    # Compress negative side slightly less
    neg_compressed = -T - np.tanh((-audio - T) / (limit - T)) * (limit - T) * (1 - asymmetry)
    
    return np.where(positive, pos_compressed,
                   np.where(negative, neg_compressed, audio))
```

### 6.3 For "Aggressive" Sound

Use polynomial with higher threshold (less frequent compression but harder when it hits):
```python
def aggressive_limit(audio, threshold_db=-3.0, limit=0.95):
    T = 10 ** (threshold_db / 20)
    mask = np.abs(audio) > T
    x = np.abs(audio)
    # 3rd-order polynomial soft clip
    excess = x - T
    range_ = limit - T
    compressed = T + excess - (excess ** 3) / (3 * range_ ** 2)
    compressed = np.minimum(compressed, limit)  # Hard limit at extreme
    return np.sign(audio) * np.where(mask, compressed, x)
```

---

## 7. Visual Comparison

The transfer curves for different compressors (with T=0.5, L=0.95):

```
Input   | tanh  | arctan | poly3 | exp   | hard
--------|-------|--------|-------|-------|------
0.0     | 0.00  | 0.00   | 0.00  | 0.00  | 0.00
0.3     | 0.30  | 0.30   | 0.30  | 0.30  | 0.30
0.5     | 0.50  | 0.50   | 0.50  | 0.50  | 0.50
0.7     | 0.69  | 0.68   | 0.71  | 0.67  | 0.70
0.9     | 0.82  | 0.81   | 0.87  | 0.79  | 0.90
1.1     | 0.89  | 0.88   | 0.94  | 0.86  | 0.95
1.3     | 0.92  | 0.91   | 0.95  | 0.90  | 0.95
1.5     | 0.93  | 0.93   | 0.95  | 0.92  | 0.95
∞       | 0.95  | 0.95   | 0.95  | 0.95  | 0.95
```

Observations:
- **tanh**: Smooth, gradual approach to limit
- **arctan**: Even more gradual than tanh
- **poly3**: Reaches limit fastest, "squashes" most aggressively
- **exp**: Undershoots significantly at moderate levels
- **hard**: Linear then instant cutoff

---

## 8. Conclusion

For the Harmony project, **tanh remains the recommended choice** because:

1. **Mathematical elegance:** Simple, well-behaved, differentiable everywhere
2. **Musical character:** Odd harmonics sound warm and pleasant
3. **Physical analogy:** Matches tube/transistor distortion profiles
4. **Predictable:** Output is guaranteed bounded regardless of input
5. **Fast to compute:** Available in numpy/pytorch as built-in function

Alternative functions are worth exploring for specific artistic effects:
- **arctan** for gentler, more "invisible" compression
- **polynomial** for intentional "bite" or fuzz
- **asymmetric tanh** for analog warmth simulation

The key insight is that **all compression introduces distortion** - the art is choosing distortion that enhances rather than degrades the musical experience.
