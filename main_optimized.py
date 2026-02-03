import numpy as np
import torch
from matplotlib import pyplot as plt
import librosa
import time
import sounddevice

# DEFINITION

default_instrument_parameters = [
    (1.0, 1.0),   # Fundamental
    (2.0, 0.5),   # 2nd harmonic
    (3.0, 0.33),  # 3rd harmonic
    (4.0, 0.25),  # 4th harmonic
    (5.0, 0.2),   # 5th harmonic
    (6.0, 0.17),  # 6th harmonic
]

def key_to_hz(key: int) -> float:
    return 440.0 * (2 ** ((key - 69) / 12))

def bin_to_hz(bin: int, num_bins: int, sample_rate: int) -> float:
    return bin * sample_rate / (2 * num_bins)

def hz_to_key(freq: float) -> int:
    return int(round(12 * np.log2(freq / 440.0) + 69))

# PRECOMPUTED MATRICES

def create_dissonance_matrix(num_keys: int, max_hz: float = 11025) -> torch.Tensor:
    """
    Precompute dissonance matrix including harmonic interactions.
    D[i,j] = total dissonance between note i and note j including all harmonics.
    """
    D = torch.zeros((num_keys, num_keys))
    
    for k1 in range(num_keys):
        for k2 in range(k1, num_keys):
            total_d = 0.0
            
            for h1, s1 in default_instrument_parameters:
                for h2, s2 in default_instrument_parameters:
                    f1 = key_to_hz(k1) * h1
                    f2 = key_to_hz(k2) * h2
                    
                    # Skip if above Nyquist frequency
                    if f1 >= max_hz or f2 >= max_hz:
                        continue
                    
                    # Skip same note with same harmonic (no self-dissonance)
                    if k1 == k2 and h1 == h2:
                        continue
                    
                    # Calculate dissonance for this pair of partials
                    x = abs(f2 - f1) / min(f1, f2)
                    d = 65 * x * np.exp(-24 * x)
                    
                    # Weight by harmonic strengths
                    total_d += d * s1 * s2
            
            D[k1, k2] = total_d
            D[k2, k1] = total_d
    
    return D

def create_interval_matrix(num_keys: int) -> torch.Tensor:
    """
    Create matrix of interval classes (0-11) between any two keys.
    Used for visualizing/debugging what intervals are most/least dissonant.
    """
    I = torch.zeros((num_keys, num_keys), dtype=torch.long)
    for i in range(num_keys):
        for j in range(num_keys):
            I[i, j] = abs(i - j) % 12
    return I

# GENERATION

def weights_to_audio(weights: torch.Tensor, sample_rate: int, duration: float) -> np.ndarray:
    """Synthesize audio using additive synthesis."""
    num_samples = int(sample_rate * duration)
    t = torch.linspace(0, duration, num_samples)
    audio = torch.zeros(num_samples)
    
    beat_duration = duration / weights.shape[1]
    
    for key in range(weights.shape[0]):
        freq = key_to_hz(key)
        for beat in range(weights.shape[1]):
            strength = weights[key, beat].item()
            if strength < 0.01:
                continue
            
            start = int(beat * beat_duration * sample_rate)
            end = int((beat + 1) * beat_duration * sample_rate)
            
            # Add fundamental + harmonics
            for h, h_str in default_instrument_parameters:
                h_freq = freq * h
                if h_freq < sample_rate / 2:
                    audio[start:end] += strength * h_str * torch.sin(2 * np.pi * h_freq * t[start:end])
    
    return audio.numpy()

# ANALYSIS

def color_weights(weights: np.ndarray) -> np.ndarray:
    (num_keys, num_beats) = weights.shape
    color_stick = np.zeros((num_keys, 3))
    for key in range(num_keys):
        color_stick[key] = plt.cm.hsv((key % 12) / 12)[:3]
    colored = np.reshape(weights, (num_keys, num_beats, 1)) * np.reshape(color_stick, (num_keys, 1, 3))
    return colored

def color_spectrogram(spectrogram: np.ndarray, sample_rate: int) -> np.ndarray:
    (num_bins, num_frames) = spectrogram.shape
    color_stick = np.zeros((num_bins, 3))
    for bin in range(num_bins):
        if bin == 0:
            continue
        freq = bin_to_hz(bin, num_bins, sample_rate)
        key = hz_to_key(freq)
        color_stick[bin] = plt.cm.hsv((key % 12) / 12)[:3]
    spectrogram = spectrogram / (spectrogram.max() + 1e-8)
    colored = np.reshape(spectrogram, (num_bins, num_frames, 1)) * np.reshape(color_stick, (num_bins, 1, 3))
    return colored

def plot_weights_and_spectrogram(title: str, weights: np.ndarray, spectrogram: np.ndarray, sample_rate: int):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)
    
    axs[0].imshow(color_weights(weights), aspect='auto')
    axs[0].set_title('Weights')
    axs[0].set_xlabel('Beats')
    axs[0].set_ylabel('MIDI Keys')
    
    axs[1].set_yscale('log')
    axs[1].imshow(color_spectrogram(spectrogram, sample_rate), aspect='auto')
    axs[1].set_title('Spectrogram')
    axs[1].set_xlabel('Time Frames')
    axs[1].set_ylabel('Frequency Bins')
    
    plt.tight_layout()
    plt.show()

my_time = time.time()

def print_time(label):
    global my_time
    new_time = time.time()
    print(f"{label}: {new_time - my_time:.4f} seconds")
    my_time = new_time

# LOSS FUNCTIONS

def calculate_loss_fast(weights: torch.Tensor, D: torch.Tensor, 
                        temporal_decay: float = 0.3,
                        target_density: float = 0.15) -> torch.Tensor:
    """
    Calculate total loss using precomputed dissonance matrix.
    
    Args:
        weights: (num_keys, num_beats) tensor
        D: (num_keys, num_keys) precomputed dissonance matrix
        temporal_decay: how much dissonance carries to next beat (0-1)
        target_density: target proportion of active notes (prevents silence)
    """
    num_keys, num_beats = weights.shape
    
    # Within-beat dissonance: sum_b (w_b^T D w_b)
    # Vectorized: trace(weights^T D weights) = sum of elementwise(D @ weights * weights)
    within_beat = torch.sum(weights * (D @ weights))
    
    # Temporal dissonance (adjacent beats): sum_b (w_b^T D w_{b+1})
    temporal = 0.0
    if num_beats > 1:
        temporal = torch.sum(weights[:, :-1] * (D @ weights[:, 1:]))
    
    # Anti-silence penalty: encourage some notes to play
    # Use smooth L1 (Huber) to avoid pushing all notes to max
    actual_density = weights.mean()
    density_penalty = 10.0 * (actual_density - target_density) ** 2
    
    # Sparsity bonus (L1 regularization) - encourages few notes rather than all quiet
    # But balanced so it doesn't drive to zero
    sparsity = weights.sum() / (num_keys * num_beats)
    sparsity_penalty = torch.abs(sparsity - target_density)
    
    total_loss = within_beat + temporal_decay * temporal + density_penalty + sparsity_penalty
    
    return total_loss

def analyze_chords(weights: torch.Tensor, threshold: float = 0.1) -> list:
    """Analyze what chords are being played at each beat."""
    chords = []
    for beat in range(weights.shape[1]):
        active = torch.where(weights[:, beat] > threshold)[0].tolist()
        # Convert to pitch classes
        pcs = sorted(set([k % 12 for k in active]))
        chords.append({
            'beat': beat,
            'active_keys': active,
            'pitch_classes': pcs,
            'num_notes': len(active)
        })
    return chords

# MAIN

def main():
    num_keys = 128
    num_beats = 8
    
    num_bins = 2048
    duration = 4.0
    sample_rate = 22050
    
    # Hyperparameters
    lr = 0.02
    steps = 200
    show_every = 25
    
    print("Precomputing dissonance matrix (this may take a moment)...")
    dissonance_matrix = create_dissonance_matrix(num_keys)
    print(f"Dissonance matrix computed: max={dissonance_matrix.max():.2f}, mean={dissonance_matrix.mean():.4f}")
    
    # Show which intervals are most dissonant
    intervals = torch.zeros(12)
    counts = torch.zeros(12)
    for i in range(num_keys):
        for j in range(i + 1, num_keys):
            interval = (j - i) % 12
            intervals[interval] += dissonance_matrix[i, j]
            counts[interval] += 1
    intervals = intervals / (counts + 1e-8)
    print("\nDissonance by interval (semitones):")
    interval_names = ['unison', 'm2', 'M2', 'm3', 'M3', 'P4', 'tritone', 'P5', 'm6', 'M6', 'm7', 'M7']
    for i, (d, name) in enumerate(zip(intervals, interval_names)):
        print(f"  {i:2d} ({name:8s}): {d:.4f}")
    
    # Initialize with gradients
    weights = torch.rand((num_keys, num_beats), requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    print(f"\nStarting optimization...")
    print_time("Start")
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Apply ReLU to ensure positive weights
        p_weights = torch.relu(weights)
        
        # Calculate loss (fast, no loops!)
        loss = calculate_loss_fast(p_weights, dissonance_matrix, 
                                   temporal_decay=0.3, target_density=0.15)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (step % show_every) == 0 or step == steps - 1 or step == 0:
            print(f"\nStep {step + 1}/{steps}, Loss: {loss.item():.4f}")
            
            # Analyze chords
            chords = analyze_chords(p_weights, threshold=0.15)
            for c in chords[:4]:  # Show first 4 beats
                print(f"  Beat {c['beat']}: {c['num_notes']} notes, pitch classes: {c['pitch_classes']}")
            
            # Generate and play audio
            audio = weights_to_audio(p_weights.detach(), sample_rate, duration)
            print_time(f"Step {step + 1} audio")
            sounddevice.play(audio, sample_rate)
            
            # Plot
            spectrogram = np.abs(librosa.stft(audio, n_fft=num_bins))
            plot_weights_and_spectrogram(f"Step {step + 1}/{steps}", p_weights.detach().numpy(), spectrogram, sample_rate)
    
    print("\nOptimization complete!")

if __name__ == "__main__":
    main()
