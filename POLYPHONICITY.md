# Training and Loss Formulations

This document explores different ways to model which notes a band member plays and at what amplitude, comparing the original "direct amplitude" approach with the newer "nonlinearity-gated" approach, and discussing the spectrum from monophonic to omniphonic instruments.

## The Core Question

We have a weight matrix `W` of shape `(num_keys, num_beats)`. What does each entry `W[k, b]` represent, and how do we turn it into sound?

## Part 1: The Semantic Spectrum of Polyphonicity

### 1.1 Monophonic (Exactly One Note)

**Definition:** The instrument can only produce one pitch at a time. Examples: human voice, saxophone, trumpet.

**Constraint:** For each beat `b`, exactly one key `k` has nonzero amplitude, OR the instrument rests (zero notes).

**Mathematical Formulation:**

```
a[k, b] ∈ {0, 1} for all k, b                    # Binary activation
Σ_k a[k, b] ∈ {0, 1} for all b                   # At most one active note
```

Where `a[k, b]` is the final activation (0 or 1).

**Training Approach - Gumbel-Softmax:**

Since argmax is not differentiable, we use the Gumbel-softmax trick:

```python
# Logits are the raw weights for this beat
logits = W[:, b]  # Shape: (num_keys,)

# Add Gumbel noise for sampling
gumbel = -log(-log(uniform(0,1)))  # Sample from Gumbel(0,1)
noisy_logits = (logits + gumbel) / temperature

# Softmax gives differentiable probabilities
soft = softmax(noisy_logits)  # Shape: (num_keys,), sums to 1

# Straight-through estimator: forward uses hard, backward uses soft
hard = one_hot(argmax(soft))
activation = hard - soft.detach() + soft  # Gradients flow through soft
```

**Loss Calculation:**

The audio amplitude for beat `b` and key `k` is:

```
Audio amplitude = a[k, b] × instrument_harmonics(k)
```

Since `a[k, b]` is effectively one-hot, the loss becomes:

```
L = Σ_b Σ_{k1, k2} D[k1, k2] × a[k1, b] × a[k2, b]
  = Σ_b a_b^T × D × a_b
```

Where `a_b` is the one-hot vector for beat `b`.

**Key Property:** Because `a_b` is one-hot, only pairs where both keys are THE SAME key contribute to loss. This means monophonic instruments only care about self-dissonance (which we usually set to zero), so they optimize purely based on temporal dissonance with other band members and adjacent beats.

### 1.2 Multiphonic (Bounded Polyphony)

**Definition:** The instrument can produce multiple pitches, but with a hard or soft limit. Examples: guitar (6 strings = up to 6 notes), string quartet (each player is monophonic but together they form a bounded polyphonic ensemble).

**Constraint:** For each beat `b`, at most `N` keys can be active, where `N` is the polyphony limit.

```
a[k, b] ∈ [0, 1] or {0, 1} for all k, b          # May be continuous or binary
Σ_k a[k, b] ≤ N for all b                        # At most N active notes
```

**The Challenge:** This is NOT a simple extension of either monophonic or fully polyphonic. It requires:
1. Selecting the top-N weights
2. Enforcing that only those N contribute to the audio
3. Keeping gradients flowing so the optimizer knows which keys to promote

**Approach A: Hard Top-K with Straight-Through**

```python
# Get top N indices (non-differentiable)
top_k_indices = topk_indices(W[:, b], k=N)

# Create hard mask
hard_mask = zeros_like(W[:, b])
hard_mask[top_k_indices] = 1.0

# For gradients, use soft top-k approximation
soft_weights = softmax(W[:, b] / temperature) * N  # Scales so sum ≈ N

# Straight-through estimator
activation = hard_mask - soft_weights.detach() + soft_weights
```

**Problem:** The hard selection of top-k is non-differentiable. The soft approximation may not accurately represent what the audio will actually be.

**Approach B: Sequential Selection (Iterative Gumbel)**

Treat as N sequential monophonic decisions, each conditioned on previous selections:

```python
activations = []
remaining_logits = W[:, b].clone()

for i in range(N):
    # Sample one note
    soft = gumbel_softmax(remaining_logits)
    hard = one_hot(argmax(soft))
    activations.append(hard)
    
    # Suppress this note for next iteration
    remaining_logits = remaining_logits - hard × large_value

a_b = sum(activations)  # May have fewer than N if some are "rest"
```

**Problem:** Complex, and the sequential nature may not match instruments that choose chords holistically.

**Approach C: Soft Threshold with L1 Regularization**

Instead of hard top-k, use L1 regularization to encourage sparsity:

```python
# No explicit top-k, just sigmoid gating
activation = sigmoid((W[:, b] - threshold) / temperature)

# Add L1 loss to encourage sparsity
sparsity_loss = lambda × sum(abs(activation))
```

**Problem:** Doesn't guarantee exactly/at most N notes, just encourages few notes.

### 1.3 Omniphonic (Unbounded Polyphony)

**Definition:** The instrument can theoretically play any or all notes simultaneously. Examples: piano, organ, synthesizer.

**Constraint:** No constraint on number of active notes.

```
a[k, b] ∈ [0, 1] or ℝ⁺ for all k, b              # Continuous activation
```

**Training Approach - Sigmoid Threshold:**

```python
# Simple sigmoid gating
activation = sigmoid((W[:, b] - threshold) / temperature)

# Or ReLU for unbounded amplitudes (original approach)
activation = relu(W[:, b])
```

**Loss Calculation:**

```
L = Σ_b Σ_{k1, k2} D[k1, k2] × a[k1, b] × a[k2, b]
  = Σ_b a_b^T × D × a_b
```

Now all pairs contribute, creating rich harmonic interactions.

## Part 2: The Amplitude Spectrum

### 2.1 Binary (On/Off)

**Definition:** Notes are either played at fixed amplitude or silent. No gradation.

```
a[k, b] ∈ {0, 1}
```

**Use Cases:**
- Pipe organ with fixed stops
- Simple synthesizer with no velocity sensitivity
- Grid-based sequencers

**Implementation:**

Use sigmoid with straight-through or hard threshold:

```python
# Forward: hard threshold
activation = (W[:, b] > threshold).float()

# Backward: sigmoid gradient (straight-through)
soft = sigmoid((W[:, b] - threshold) / temperature)
activation = activation - soft.detach() + soft
```

### 2.2 Continuous (Arbitrary Amplitudes)

**Definition:** Notes can have any amplitude in a continuous range, typically [0, 1] or [0, ∞).

```
a[k, b] ∈ [0, 1]  # Bounded
# OR
a[k, b] ∈ [0, ∞)  # Unbounded (ReLU)
```

**Use Cases:**
- Piano with velocity-sensitive keys
- String instruments with varying bow pressure
- Any realistic physical instrument

**Implementation:**

```python
# Bounded [0, 1] with sigmoid
activation = sigmoid(W[:, b])

# Bounded [0, 1] with learnable threshold
activation = sigmoid((W[:, b] - threshold) / temperature)

# Unbounded positive with ReLU
activation = relu(W[:, b])

# Unbounded positive with softplus (smooth ReLU)
activation = softplus(W[:, b]) = log(1 + exp(W[:, b]))
```

## Part 3: Direct Physical Weights vs. Nonlinearity-Gated Weights

### 3.1 Original Approach: Weights ARE Amplitudes

In `main.py` and `main_optimized.py`, weights directly represent physical amplitudes:

```python
# Initialize weights that will be optimized
W = torch.rand((num_keys, num_beats), requires_grad=True)

# Enforce positivity
positive_W = relu(W)

# Synthesis - weights directly scale the sine waves
audio += positive_W[k, b] × harmonic_strength[h] × sin(2π × freq × t)

# Loss - weights directly multiply dissonance
L = Σ_b Σ_{k1,k2} D[k1, k2] × positive_W[k1, b] × positive_W[k2, b]
```

**Characteristics:**
- **Regression problem:** Optimizer learns actual amplitude values
- **Physical interpretation:** W[k, b] = "how loud is key k on beat b"
- **Smooth gradients:** No discontinuities in the forward pass
- **Unbounded polyphony:** All keys can have any amplitude
- **Continuous amplitudes:** ReLU allows any positive value

**Advantages:**
1. Simple, direct interpretation
2. Smooth loss landscape (no sampling noise)
3. Gradients flow directly: ∂L/∂W is continuous
4. Easy to regularize (L2 on weights = discourage loud notes)

**Disadvantages:**
1. No easy way to enforce sparsity (L1 helps but doesn't guarantee)
2. No easy way to enforce monophonic constraint
3. May converge to "muddy" solutions with many quiet notes

### 3.2 Current Approach: Weights ARE Neural Network Parameters

In `harmony/band.py`, weights go through nonlinearities to produce activations:

**PolyphonicMember:**
```python
# Weights are parameters
self.weights = Parameter(torch.randn(num_keys, num_beats) × 0.1)

# Sigmoid gating produces activation
activation = sigmoid((weights - threshold) / temperature)

# Audio uses the activation, not raw weights
audio += activation[k, b] × harmonic_strength[h] × sin(...)
```

**MonophonicMember:**
```python
# Gumbel-softmax produces one-hot (or nearly one-hot)
activation = gumbel_softmax(weights[:, b] / temperature)
# Then straight-through estimator...
```

**Characteristics:**
- **Classification problem (monophonic):** Weights represent preferences/logits, not amplitudes
- **Gating problem (polyphonic):** Weights control gates, not direct amplitude
- **Discontinuous forward pass:** Hard threshold/argmax create discontinuities
- **Requires straight-through:** Gradients flow through soft approximation

**Advantages:**
1. Can enforce hard constraints (monophonic, binary on/off)
2. Natural sparsity with thresholding
3. Matches discrete nature of real instruments (key is pressed or not)

**Disadvantages:**
1. Noisy gradients from Gumbel sampling (can be reduced by lowering temperature)
2. Bias from straight-through estimator (forward uses hard, backward uses soft)
3. Temperature tuning is tricky (low = accurate but high variance gradients)

## Part 4: The Multiphonic Challenge - Why It's Hard

The fundamental issue with multiphonic (bounded polyphony) is that it sits in an awkward middle ground:

### Why Monophonic Works Well

With exactly one note, the problem reduces to classification: which of K keys?

```
Probability distribution over K choices → Gumbel-softmax works well
```

### Why Omniphonic Works Well

With unlimited notes, each key's activation is independent:

```
Each key k has independent activation → Sigmoid per key works well
```

### Why Multiphonic Is Hard

With N-of-K selection, keys are NOT independent:

```
P(k1 active AND k2 active) ≠ P(k1 active) × P(k2 active)
```

They compete for the "slots." This creates a combinatorial problem:

```
Number of valid N-of-K combinations = C(K, N) = K! / (N!(K-N)!)
```

For a guitar (6 strings, 20 frets = ~40 possible pitches, choose up to 6):
```
C(40, 6) ≈ 3.8 million possible chords
```

**The Training Problem:**

The optimizer needs to:
1. Decide WHICH combination of N notes is best
2. Learn the optimal amplitudes for those N notes

This is a discrete selection problem (which N notes?) combined with a continuous optimization (what amplitudes?).

### Possible Solutions

**Solution 1: Factorized Decision**

Treat as N independent monophonic decisions:

```python
for string in range(6):
    # Each string independently chooses a note (or rests)
    note[string] = gumbel_softmax(logits_per_string[string])
```

This is how real guitars work! Each string is monophonic (one fret or open), and 6 monophonic strings = polyphonic chord.

**Solution 2: Top-K with Differentiable Sort**

Use recent research on differentiable sorting/top-k:

```python
# Soft top-k using sinkhorn or similar
soft_top_k = differentiable_topk(W[:, b], k=N)
activation = soft_top_k
```

**Solution 3: Curriculum Learning**

Start with omniphonic (easy), gradually increase sparsity:

```python
# Early training: sigmoid gating (omniphonic)
# Late training: progressively increase threshold/temperature
activation = sigmoid((W[:, b] - threshold(epoch)) / temperature(epoch))
```

## Part 5: Recommended Formulations by Use Case

### Piano (Omniphonic + Continuous)

```python
# Direct amplitude approach (original)
W = Parameter(torch.randn(num_keys, num_beats))
a = relu(W)  # Positive amplitudes

# Or with bounded amplitude
a = sigmoid(W)  # [0, 1]
```

Loss: `L = Σ_b a_b^T × D × a_b + regularization`

### Guitar (Multiphonic N=6 + Continuous/Fixed)

```python
# Factorized by string (each string is monophonic)
strings = []
for s in range(6):
    string_logits = W[s, :, b]  # Frets for this string on this beat
    string_note = gumbel_softmax(string_logits)
    strings.append(string_note)

# Sum all string activations (max 6 notes)
a = sum(strings)
```

### Trumpet (Monophonic + Continuous Amplitude via Breath)

```python
# Single note selection (discrete)
note = gumbel_softmax(W[:, b])  # One-hot

# Amplitude from separate breath parameter
amplitude = sigmoid(breath[b])

a = amplitude × note
```

### Organ (Omniphonic + Binary per Stop)

```python
# Multiple stops (flute, reed, etc.), each is binary on/off
for stop in stops:
    active = sigmoid((W[stop, b] - threshold) / temp)
    # Straight-through for binary
    hard = (W[stop, b] > threshold).float()
    activation = hard - active.detach() + active
```

## Part 6: Summary Table

| Formulation | Polyphony | Amplitude | Forward Pass | Training | Best For |
|-------------|-----------|-----------|--------------|----------|----------|
| Direct ReLU | Omniphonic | Continuous [0,∞) | `a = relu(W)` | Direct regression | Piano, synth pads |
| Sigmoid Gate | Omniphonic | Continuous [0,1] | `a = sigmoid(W)` | Direct regression | Velocity-sensitive keys |
| Threshold Binary | Omniphonic | Binary {0,1} | `a = (W > θ).float()` | Straight-through sigmoid | Pipe organ stops |
| Gumbel One-Hot | Monophonic | Fixed (on/off) | `a = one_hot(argmax(W))` | Gumbel-softmax + ST | Voice, sax, lead instruments |
| Factorized Strings | Multiphonic N | Continuous | `a = Σ_{i=1}^N gumbel(W_i)` | N× Gumbel-softmax | Guitar, harp |
| Top-K Hard | Multiphonic N | Binary | `a = topk_mask(W, k=N)` | Straight-through soft top-k | Chord voicing selection |

## Part 7: The Death of the Piano (Debugging Notes)

In the current implementation, the piano dies after ~20 optimization steps with higher learning rates. The images show:
- `100_steps_2e-2_lr.png`: Piano completely silent (all black)
- `10_steps_1e-2_lr.png`: Piano has activity

**Hypothesis:** The sigmoid gating with threshold creates a "dead zone." If all weights fall below the threshold, sigmoid outputs ~0, and gradients become vanishingly small (sigmoid' = sigmoid × (1-sigmoid) ≈ 0).

**Original approach advantage:** ReLU doesn't have this problem - negative weights have gradient 0, but positive weights have gradient 1, so there's always a path for recovery.

**Solution:** Either:
1. Use ReLU instead of sigmoid for the polyphonic case (original approach)
2. Use a learnable threshold that adapts during training
3. Add a "resurrection" mechanism that reinitializes dead weights

## Appendix: Mathematical Notation Reference

- `W`: Raw weight tensor `(num_keys, num_beats)` — what PyTorch optimizes
- `a`: Activation tensor `(num_keys, num_beats)` — what actually produces sound
- `D`: Dissonance matrix `(num_keys, num_keys)` — precomputed
- `a_b`: Activation vector for beat `b` `(num_keys,)`
- `gumbel_softmax(logits, τ)`: Differentiable sampling where temperature `τ` controls discreteness
- `ST(x_hard, x_soft)`: Straight-through estimator: `x_hard - x_soft.detach() + x_soft`
