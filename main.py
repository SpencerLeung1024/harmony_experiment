import numpy as np
import torch
from matplotlib import pyplot as plt
import librosa
import time
import sounddevice

# DEFINITION

default_instrument_parameters = [
    (1.0, 1.0), # Fundamental has full strength
    (2.0, 0.5), # Second harmonic has half strength
    (3.0, 0.33),
    (4.0, 0.25),
    (5.0, 0.2),
    (6.0, 0.17)
]

def key_to_hz(key: int) -> float:
    return 440.0 * (2 ** ((key - 69) / 12))

def bin_to_hz(bin: int, num_bins: int, sample_rate: int) -> float:
    return bin * sample_rate / (2 * num_bins)

def hz_to_key(freq: float) -> int:
    return int(round(12 * np.log2(freq / 440.0) + 69))

# GENERATION

# Use additive synthesis instead of regenerating audio from a spectrogram
def weights_to_audio(weights: torch.Tensor, sample_rate: int, duration: float) -> np.ndarray:
    """Synthesize audio directly - ~100x faster than Griffin-Lim."""
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
            
            # Add fundamental + harmonics directly
            for h, h_str in default_instrument_parameters:
                h_freq = freq * h
                if h_freq < sample_rate / 2:
                    audio[start:end] += strength * h_str * torch.sin(2 * np.pi * h_freq * t[start:end])
    
    return audio.numpy()

# ANALYSIS

# Use C = red, C# = orange, etc.
def color_weights(weights: np.ndarray) -> np.ndarray:
    (num_keys, num_beats) = weights.shape
    # If weights has height 128, generate a (128, 1, 3) color stick
    # Note: key 0 is always C-1
    color_stick = np.zeros((num_keys, 3))
    for key in range(num_keys):
        color_stick[key] = plt.cm.hsv((key % 12) / 12)[:3]
    # Element-wise multiply weights with color stick
    colored = np.reshape(weights, (num_keys, num_beats, 1)) * np.reshape(color_stick, (num_keys, 1, 3))
    return colored

def color_spectrogram(spectrogram: np.ndarray, sample_rate: int) -> np.ndarray:
    (num_bins, num_frames) = spectrogram.shape
    color_stick = np.zeros((num_bins, 3))
    for bin in range(num_bins):
        # The first bin is 0 Hz which doesn't map to a key, so just leave it black
        if bin == 0:
            continue
        # bin -> hz -> key
        freq = bin_to_hz(bin, num_bins, sample_rate)
        key = hz_to_key(freq)
        color_stick[bin] = plt.cm.hsv((key % 12) / 12)[:3]
    # Element-wise multiply spectrogram with color stick
    spectrogram = spectrogram / spectrogram.max() # Floats must be in [0.0, 1.0]
    colored = np.reshape(spectrogram, (num_bins, num_frames, 1)) * np.reshape(color_stick, (num_bins, 1, 3))
    return colored

def plot_weights_and_spectrogram(title: str, weights: np.ndarray, spectrogram: np.ndarray, sample_rate: int):
    fig, axs = plt.subplots(1, 2)

    fig.suptitle(title)

    #axs[0].imshow(weights.detach().numpy(), aspect='auto', cmap='gray')
    axs[0].imshow(color_weights(weights.detach().numpy()), aspect='auto')
    axs[0].set_title('Weights')
    axs[0].set_xlabel('Beats')
    axs[0].set_ylabel('MIDI Keys')

    axs[1].set_yscale('log')
    #axs[1].imshow(spectrogram, aspect='auto', cmap='gray')
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

# MAIN

def main():
    num_keys = 128
    num_beats = 8

    num_bins = 2048

    duration = 4.0
    sample_rate = 22050

    # Hyperparameters
    lr = 0.01
    steps = 100
    show_every = 20

    # Initialize with gradients
    weights = torch.rand((num_keys, num_beats), requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        print(f"Step {step + 1}/{steps}")

        # Negative amplitudes don't make sense so enforce positive
        p_weights = torch.relu(weights)
        # Note: clamping or otherwise directly modifying weights breaks things so just make negative weights do nothing
        # Outside of this loop, "weights" is always p_weights post-ReLU
        
        # Optimize
        # Calculate loss in PyTorch (not numpy!)
        loss = torch.tensor(0.0, requires_grad=True)
        for beat in range(p_weights.shape[1]):
            for k1 in range(p_weights.shape[0]):
                for k2 in range(k1 + 1, p_weights.shape[0]):
                    f1, f2 = key_to_hz(k1), key_to_hz(k2)
                    x = abs(f2 - f1) / min(f1, f2)
                    # Reaches a maximum at (0.041667, 0.99634)
                    dissonance = 65 * x * torch.exp(torch.tensor([-24 * x]))
                    loss = loss + dissonance * p_weights[k1, beat] * p_weights[k2, beat]
        
        loss.backward()
        print_time("Calculated loss")
        print("Loss:", loss.item())
        optimizer.step()
        print_time("Updated weights")

        if (step % show_every) == 0 or step == steps - 1:
            # Play audio
            audio = weights_to_audio(p_weights, sample_rate, duration)
            print_time("Generated audio")
            sounddevice.play(audio, sample_rate)

            # Plot
            spectrogram = librosa.stft(audio, n_fft=num_bins)
            spectrogram = np.abs(spectrogram)
            print_time("Generated spectrogram")
            plot_weights_and_spectrogram(f"Step {step + 1}/{steps}", p_weights, spectrogram, sample_rate)

if __name__ == "__main__":
    main()
