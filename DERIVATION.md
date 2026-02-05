# Derivation of the Dissonance Matrix

This document explains how the optimized loss calculation `torch.sum(weights * (D @ weights))` was derived from the original sextuple loop.

## Original Specification

The original loss was defined as:

```python
loss = 0
for beat in range(num_beats):                    # Loop 1
    for k1 in range(num_keys):                   # Loop 2
        for k2 in range(k1 + 1, num_keys):       # Loop 3 (pairs of keys)
            for h1, s1 in harmonics:             # Loop 4
                for h2, s2 in harmonics:         # Loop 5
                    f1 = key_to_hz(k1) * h1
                    f2 = key_to_hz(k2) * h2
                    x = abs(f2 - f1) / min(f1, f2)
                    d = 65 * x * exp(-24 * x)    # Two-tone dissonance
                    loss += d * s1 * s2 * weights[k1, beat] * weights[k2, beat]
```

Plus temporal dissonance:

```python
for beat1 in range(num_beats):                        # Loop 0 (current beat)
    for beat2 in range(max(beat1 - 1, 0), beat1 + 1): # Loop 1 (up to 1 previous beat and the beat itself)
        for k1 in range(num_keys):
            for k2 in range(num_keys):
                # ... same calculation for adjacent beats
```

## Step 1: Recognize Independence

**Key insight:** The dissonance between two notes depends only on:
1. Their MIDI key numbers (`k1`, `k2`)
2. The instrument harmonics (fixed)

It does **NOT** depend on:
- Which beat they're on
- The actual weight values
- Temporal context

## Step 2: Precompute the Dissonance Matrix

Since dissonance is beat-independent, we can compute `D[k1, k2]` once:

```python
D = zeros((num_keys, num_keys))

for k1 in range(num_keys):
    for k2 in range(k1, num_keys):  # Upper triangle + diagonal
        total_d = 0
        
        for h1, s1 in harmonics:
            for h2, s2 in harmonics:
                f1 = key_to_hz(k1) * h1
                f2 = key_to_hz(k2) * h2
                
                if k1 == k2 and h1 == h2:
                    continue  # No self-dissonance
                
                x = abs(f2 - f1) / min(f1, f2)
                d = 65 * x * exp(-24 * x)
                total_d += d * s1 * s2
        
        D[k1, k2] = total_d
        D[k2, k1] = total_d  # Symmetric
```

**Complexity:** This is done once at startup: O(keys² × harmonics²) ≈ 128² × 36 ≈ 590k operations.

## Step 3: Express Loss as Matrix Operations

### Within-Beat Dissonance

For a single beat `b`, the loss is:

```
loss_b = Σ_{k1,k2} D[k1, k2] × w[k1, b] × w[k2, b]
```

This is a **quadratic form**: `w_b^T × D × w_b`

Where:
- `w_b` is a column vector of shape (128, 1) — weights for all keys at beat b
- `D` is the matrix (128, 128)
- Result is a scalar

### Vectorizing Across All Beats

Let `W` be the full weight matrix of shape (128, 8).

`D @ W` gives us `(128, 128) @ (128, 8) = (128, 8)`

Each column of the result is `D @ w_b`.

Now, element-wise multiply `W * (D @ W)` and sum:

```
torch.sum(W * (D @ W))
  = Σ_b Σ_k W[k, b] × (D @ W)[k, b]
  = Σ_b Σ_k W[k, b] × Σ_j D[k, j] × W[j, b]
  = Σ_b Σ_k Σ_j D[k, j] × W[k, b] × W[j, b]
  = Σ_b w_b^T × D × w_b
```

This is exactly the within-beat loss!

### Temporal Dissonance

For adjacent beats, we want:

```
temporal = Σ_b w_b^T × D × w_{b+1}
```

Using matrix notation:
- `W[:, :-1]` is all beats except the last: shape (128, 7)
- `W[:, 1:]` is all beats except the first: shape (128, 7)

```
D @ W[:, 1:]` gives `(128, 128) @ (128, 7) = (128, 7)

Each column is D @ w_{b+1}

Element-wise multiply with W[:, :-1]:
  W[:, :-1] * (D @ W[:, 1:])
  
Sum gives: Σ_b Σ_k W[k, b] × (D @ w_{b+1})[k]
         = Σ_b Σ_k Σ_j W[k, b] × D[k, j] × W[j, b+1]
         = Σ_b w_b^T × D × w_{b+1}
```

## Step 4: Final Implementation

```python
def calculate_loss_fast(weights, D, temporal_decay=0.3):
    # Within-beat: Σ_b w_b^T D w_b
    within_beat = torch.sum(weights * (D @ weights))
    
    # Temporal: Σ_b w_b^T D w_{b+1}
    temporal = torch.sum(weights[:, :-1] * (D @ weights[:, 1:]))
    
    return within_beat + temporal_decay * temporal
```

## Complexity Comparison

| Operation | Original (Loops) | Optimized (Matrix) |
|-----------|-----------------|-------------------|
| Setup | None | ~590k ops (once) |
| Per-step | O(beats × keys²) = 8 × 128² ≈ 131k | O(keys² × beats) = same |
| Per-step (with harmonics) | O(beats × keys² × harm²) ≈ 4.7M | O(keys² × beats) = 131k |
| Memory | O(beats × keys) for W | O(keys²) for D |

The key win: **harmonics are in the precomputation**, not the per-step calculation.

## Why This Works Mathematically

The dissonance function has the property:

```
dissonance(f1, f2) = f(ratio) where ratio = f2/f1
```

For two notes at MIDI keys k1 and k2:
```
frequency(k) = 440 × 2^((k-69)/12)
ratio = 2^((k2-k1)/12)
```

The ratio depends **only on the interval**, not the absolute pitch.

## Summary

1. **Precompute** `D[i,j]` = dissonance between notes i and j (including harmonics)
2. **Express** within-beat loss as `Σ_b w_b^T D w_b` = `torch.sum(W * (D @ W))`
3. **Express** temporal loss as `Σ_b w_b^T D w_{b+1}` = `torch.sum(W[:,:-1] * (D @ W[:,1:]))`
4. **Result**: Same loss function, ~36× faster per step (harmonics precomputed)