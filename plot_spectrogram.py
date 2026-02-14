# Simple script to plot a spectrogram

import numpy as np
import sys
from matplotlib import pyplot as plt
import librosa

from color_service import ColorService

NUM_BINS = 4096

def hz_to_bin(freq: float, num_bins: int, sample_rate: int) -> int:
    return int((freq / (sample_rate / 2)) * num_bins)

def plot_spectrogram(audio: np.ndarray, sample_rate: int, title_suffix: str = "", interactive: bool = True):
    spectrogram = np.abs(librosa.stft(audio, n_fft=NUM_BINS))

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    colormap = ColorService.color_spectrogram(spectrogram, sample_rate)

    # Spectrogram plot with Hz ticks
    ax.imshow(colormap, aspect='auto')
    title = 'Spectrogram'
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title)
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_yscale('log')
    
    # Choose Hz ticks:
    hz_ticks = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    num_bins = spectrogram.shape[0]
    bin_indices = [hz_to_bin(hz, num_bins, sample_rate) for hz in hz_ticks]
    
    ax.set_yticks(bin_indices)
    ax.set_yticklabels([f"{hz}" for hz in hz_ticks])

    # Fix height
    ax.set_ylim(bottom=hz_to_bin(hz_ticks[-1], num_bins, sample_rate), top=hz_to_bin(hz_ticks[0], num_bins, sample_rate))

    plt.tight_layout()
    return fig

def main(audio_path: str, spectrogram_path: str):
    audio, sample_rate = librosa.load(audio_path, sr=None)
    fig = plot_spectrogram(audio, sample_rate, title_suffix=audio_path)
    fig.savefig(spectrogram_path)
    #plt.show() # Not interactive

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_spectrogram.py <audio_path> <spectrogram_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    spectrogram_path = sys.argv[2]
    main(audio_path, spectrogram_path)
