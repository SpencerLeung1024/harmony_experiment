import numpy as np
import torch
from matplotlib import pyplot as plt
import librosa
import time

# CONVERSION

def key_to_hz(key: int) -> float:
    """Convert MIDI key number to frequency in Hz."""
    return 440.0 * (2 ** ((key - 69) / 12))

def hz_to_bin(freq: float, num_bins: int, sample_rate: int = 44100) -> int:
    """Convert frequency in Hz to spectrogram bin index."""
    return int((freq / (sample_rate / 2)) * (num_bins - 1))

# GENERATION

def default_instrument(spectrogram: np.ndarray, num_beats: int, key: int, beat: int, strength: float):
    #print("input", key, beat, strength)
    instrument_parameters = [
        (1.0, 1.0), # Fundamental has full strength
        (2.0, 0.5), # Second harmonic has half strength
        (3.0, 0.33),
        (4.0, 0.25),
        (5.0, 0.2),
        (6.0, 0.17)
    ]
    # First, get the start and end times in the spectrogram for the given beat
    start_time = int((beat / num_beats) * spectrogram.shape[1])
    end_time = int(((beat + 1) / num_beats) * spectrogram.shape[1])
    # Ensure in bounds of spectrogram
    start_time = max(0, min(spectrogram.shape[1] - 1, start_time))
    end_time = max(0, min(spectrogram.shape[1], end_time))
    # First, get the frequency in Hz for the given key
    freq = key_to_hz(key)
    for harmonic, relative_strength in instrument_parameters:
        harmonic_freq = freq * harmonic
        # Then, get the corresponding bin index in the spectrogram
        bin = hz_to_bin(harmonic_freq, spectrogram.shape[0])
        # Ensure in bounds of spectrogram
        if bin < 0 or bin >= spectrogram.shape[0]:
            continue
        # For each time in [start_time, end_time), add the strength to the spectrogram
        for t in range(start_time, end_time):
            harmonic_strength = strength * relative_strength
            #print(bin, t, harmonic_strength)
            spectrogram[bin, t] += harmonic_strength

def generate_weights(num_keys: int, num_beats: int, avg_value: float) -> torch.Tensor:
    weights = torch.rand((num_keys, num_beats))
    # Rescale to have the desired average value
    weights *= (avg_value / weights.mean())
    return weights

def weights_to_spectrogram(weights: torch.Tensor, num_bins: int, num_times: int,instrument: callable) -> np.ndarray:
    spectrogram = np.zeros((num_bins, num_times))
    # For each weight, apply the instrument function to the spectrogram
    for k in range(weights.shape[0]):
        for b in range(weights.shape[1]):
            strength = weights[k, b].item()
            instrument(spectrogram, weights.shape[1], k, b, strength)
    print(spectrogram.mean())
    return spectrogram

def spectrogram_to_audio(spectrogram: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
    """Convert spectrogram to audio waveform using inverse STFT."""
    # Use librosa's Griffin-Lim algorithm for phase reconstruction
    audio = librosa.griffinlim(spectrogram)
    return audio

# LEARNING

def calculate_two_tones_loss(freq1: float, freq2: float) -> float:
    # Enforce freq1 <= freq2
    if freq2 > freq1:
        freq1, freq2 = freq2, freq1
    x = (freq2 - freq1) / freq1 # 0 is unison, 1 is octave, etc.
    # Reaches a maximum at (0.041667, 0.99634)
    return 65 * x * np.exp(-24 * x)

#def calculate_spectrogram_loss
# ???

# USAGE

def plot_weights_and_spectrogram(weights: torch.Tensor, spectrogram: np.ndarray):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].imshow(weights.numpy(), aspect='auto', cmap='gray')
    axs[0].set_title('Weights')
    axs[0].set_xlabel('Beats')
    axs[0].set_ylabel('MIDI Keys')

    axs[1].set_yscale('log')
    axs[1].imshow(spectrogram, aspect='auto', cmap='gray')
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

def main():
    num_keys = 128
    num_beats = 128
    avg_value = 0.25

    num_bins = 8192
    num_times = 1024

    # Hyperparameters
    lr = 0.01
    steps = 20

    weights = generate_weights(num_keys, num_beats, avg_value)
    print_time("Generated weights")
    for step in range(steps):
        print(f"Step {step + 1}/{steps}")
        spectrogram = weights_to_spectrogram(weights, num_bins, num_times, default_instrument)
        print_time("Generated spectrogram")
        #loss = calculate_loss(spectrogram)
        # Optimize
        # ???
        audio = spectrogram_to_audio(spectrogram)
        print_time("Generated audio")
        # Play audio
        # ???
        plot_weights_and_spectrogram(weights, spectrogram)
        # Wait for click
        plt.waitforbuttonpress()

if __name__ == "__main__":
    main()
