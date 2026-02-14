"""Test script to evaluate voice synthesis in isolation."""

import numpy as np
import torch
from scipy.io import wavfile
from matplotlib import pyplot as plt
import librosa

from instruments import get_instrument, ADSR, VoiceInstrument
from audio_service import AudioService
from color_service import ColorService

SAMPLE_RATE = 22050
OUTPUT_FOLDER = "output"

def hz_to_bin(freq: float, num_bins: int, sample_rate: int) -> int:
    return int((freq / (sample_rate / 2)) * num_bins)

def test_single_note(instrument_name: str, note_freq: float, duration: float = 1.0, velocity: float = 0.5):
    """Test a single note with the voice instrument."""
    instrument = get_instrument(instrument_name)
    
    # Generate sound
    audio = instrument.get_sound(note_freq, velocity, duration, SAMPLE_RATE)
    
    # Apply limiter
    audio = AudioService.apply_limiter(audio)
    
    return audio

def plot_voice_spectrogram(audio: np.ndarray, sample_rate: int, title: str, filename: str):
    """Plot and save a spectrogram for voice analysis."""
    NUM_BINS = 4096
    spectrogram = np.abs(librosa.stft(audio, n_fft=NUM_BINS))
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Use log scale for amplitude to see quieter harmonics
    log_spectrogram = np.log1p(spectrogram)
    
    colormap = ColorService.color_spectrogram(spectrogram, sample_rate)
    
    ax.imshow(colormap, aspect='auto', origin='lower')
    ax.set_title(title)
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_yscale('log')
    
    # Hz ticks
    hz_ticks = [50, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000, 3000, 5000, 10000]
    num_bins = spectrogram.shape[0]
    bin_indices = [min(hz_to_bin(hz, num_bins, sample_rate), num_bins - 1) for hz in hz_ticks]
    
    ax.set_yticks(bin_indices)
    ax.set_yticklabels([f"{hz}" for hz in hz_ticks])
    ax.set_ylim(hz_to_bin(50, num_bins, sample_rate), hz_to_bin(4000, num_bins, sample_rate))
    
    plt.tight_layout()
    fig.savefig(filename)
    plt.close()
    
    return fig

def test_scale(instrument_name: str, output_prefix: str):
    """Test a major scale like the user's recording."""
    # A major scale from A2 to A3 (like user's recording)
    # A2 = 110 Hz, B2 = 123.47, C#3 = 138.59, D3 = 146.83, E3 = 164.81, F#3 = 185.00, G#3 = 207.65, A3 = 220
    scale_freqs = [110.00, 123.47, 138.59, 146.83, 164.81, 185.00, 207.65, 220.00]
    note_names = ['A2', 'B2', 'C#3', 'D3', 'E3', 'F#3', 'G#3', 'A3']
    
    note_duration = 0.5  # half second per note
    gap = 0.1  # 100ms gap between notes
    
    full_audio = []
    
    for freq, name in zip(scale_freqs, note_names):
        print(f"Generating {name} ({freq:.2f} Hz)...")
        note_audio = test_single_note(instrument_name, freq, note_duration, velocity=0.6)
        full_audio.append(note_audio)
        # Add silence
        full_audio.append(np.zeros(int(gap * SAMPLE_RATE)))
    
    audio = np.concatenate(full_audio)
    
    # Save audio
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(f"{OUTPUT_FOLDER}/{output_prefix}_scale.wav", SAMPLE_RATE, audio_int16)
    
    # Plot spectrogram
    plot_voice_spectrogram(audio, SAMPLE_RATE, 
                          f"{instrument_name} Scale (A2-A3)", 
                          f"{OUTPUT_FOLDER}/{output_prefix}_spectrogram.png")
    
    print(f"Saved to {OUTPUT_FOLDER}/{output_prefix}_scale.wav and {output_prefix}_spectrogram.png")
    
    return audio

def compare_formant_shift():
    """Compare voice with and without formant shifting."""
    # Create two instruments
    base_formants = [
        (750, 1.00, 85),
        (1150, 0.70, 110),
        (2650, 0.35, 160),
        (3500, 0.18, 220),
    ]
    
    # Without formant shift
    voice_fixed = VoiceInstrument(
        formants=base_formants,
        adsr=ADSR(attack=0.15, decay=0.10, sustain=0.85, release=0.50),
        num_harmonics=24,
        vibrato_rate=5.5,
        vibrato_depth=0.02,
        breath_amount=0.15,
        inharmonicity=0.0010,
        formant_shift_rate=0.0  # No shift
    )
    
    # With formant shift
    voice_shifted = VoiceInstrument(
        formants=base_formants,
        adsr=ADSR(attack=0.15, decay=0.10, sustain=0.85, release=0.50),
        num_harmonics=24,
        vibrato_rate=5.5,
        vibrato_depth=0.02,
        breath_amount=0.15,
        inharmonicity=0.0010,
        formant_shift_rate=0.15  # 15% per octave
    )
    
    # Test at low and high pitch
    low_freq = 110.0   # A2
    high_freq = 440.0  # A4
    
    print("Testing formant shift effect...")
    
    # Generate test tones
    for label, freq in [("A2", low_freq), ("A4", high_freq)]:
        print(f"  Testing {label} ({freq} Hz)...")
        
        audio_fixed = voice_fixed.get_sound(freq, 0.5, 1.0, SAMPLE_RATE)
        audio_shifted = voice_shifted.get_sound(freq, 0.5, 1.0, SAMPLE_RATE)
        
        audio_fixed = AudioService.apply_limiter(audio_fixed)
        audio_shifted = AudioService.apply_limiter(audio_shifted)
        
        # Save
        wavfile.write(f"{OUTPUT_FOLDER}/voice_fixed_{label}.wav", SAMPLE_RATE, 
                     (audio_fixed * 32767).astype(np.int16))
        wavfile.write(f"{OUTPUT_FOLDER}/voice_shifted_{label}.wav", SAMPLE_RATE,
                     (audio_shifted * 32767).astype(np.int16))
        
        # Plot
        plot_voice_spectrogram(audio_fixed, SAMPLE_RATE,
                              f"Fixed Formants - {label}",
                              f"{OUTPUT_FOLDER}/voice_fixed_{label}_spec.png")
        plot_voice_spectrogram(audio_shifted, SAMPLE_RATE,
                              f"Shifted Formants - {label}",
                              f"{OUTPUT_FOLDER}/voice_shifted_{label}_spec.png")

if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("=" * 60)
    print("Voice Instrument Test Suite")
    print("=" * 60)
    
    # Test 1: Scale like user's recording
    print("\n[Test 1] A2-A3 scale comparison")
    print("-" * 40)
    test_scale("voice_aah", "voice_aah")
    
    # Test 2: Compare formant shifting
    print("\n[Test 2] Formant shift comparison")
    print("-" * 40)
    compare_formant_shift()
    
    print("\n" + "=" * 60)
    print("Tests complete! Check output/ folder for results.")
    print("=" * 60)
