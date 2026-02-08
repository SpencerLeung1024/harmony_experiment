# TODO: Make this do more than just a short test

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import librosa
import sounddevice

from song import Song
from members import get_member
from audio_service import AudioService
from color_service import ColorService

OUTPUT_FOLDER = "output"
SAMPLE_RATE = 22050
NUM_BINS = 4096

def hz_to_bin(freq: float, num_bins: int, sample_rate: int) -> int:
    return int((freq / (sample_rate / 2)) * num_bins)

# from v2\harmony\visualization.py
def save_audio(
    audio: np.ndarray,
    filename: str,
    sample_rate: int,
):
    # Normalize to int16 range
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, audio_int16)

def plot_weights(song: Song):
    fig, axs = plt.subplots(2, 2)
    for member_idx, (axx, axy) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        member = song.members[member_idx]
        this_ax = axs[axx, axy]

        colormap = ColorService.color_weights(member)

        this_ax.imshow(colormap, aspect="auto")
        this_ax.set_title(f"{member.name} Weights")
        this_ax.set_xlabel("Note Index")
        this_ax.set_ylabel("Key Index")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/weights.png")
    plt.show()

def plot_spectrogram(audio: np.ndarray, sample_rate: int):
    spectrogram = np.abs(librosa.stft(audio, n_fft=NUM_BINS))

    fig, ax = plt.subplots(1, 1)

    colormap = ColorService.color_spectrogram(spectrogram, sample_rate)

    # Spectrogram plot with Hz ticks
    ax.imshow(colormap, aspect='auto')
    ax.set_title('Spectrogram')
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
    #print(axs[1].get_ylim())
    # (11220.18454301963, 4.0)
    ax.set_ylim(bottom=hz_to_bin(hz_ticks[-1], num_bins, sample_rate), top=hz_to_bin(hz_ticks[0], num_bins, sample_rate))
    #print(axs[1].get_ylim())
    # (929.0, 4.0)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/spectrogram.png")
    plt.show()

def main():
    # Create a song
    song = Song(
        measures=8,
        tempo=120,
        beats_per_measure=4,
        ticks_per_beat=6,
        sample_rate=SAMPLE_RATE
    )

    # Create members
    tick_duration = song.tick_duration()
    total_ticks = song.total_ticks()
    song.members.extend([
        get_member(
            "piano",
            song=song,
            tick_duration=tick_duration,
            total_ticks=total_ticks,
            ticks_per_note=24
        ),
        get_member(
            "guitar",
            song=song,
            tick_duration=tick_duration,
            total_ticks=total_ticks,
            ticks_per_note=2
        ),
        get_member(
            "bass",
            song=song,
            tick_duration=tick_duration,
            total_ticks=total_ticks,
            ticks_per_note=6
        ),
        get_member(
            "synth",
            song=song,
            tick_duration=tick_duration,
            total_ticks=total_ticks,
            ticks_per_note=1
        )
    ])

    # TODO: Create LossHandler and OptimHandler

    # Render and play the audio
    audio = AudioService.render(song)
    save_audio(audio, f"{OUTPUT_FOLDER}/audio.wav", SAMPLE_RATE)
    sounddevice.play(audio, SAMPLE_RATE)

    # Plot the weights
    plot_weights(song)

    # Plot the spectrogram
    plot_spectrogram(audio, SAMPLE_RATE)


if __name__ == "__main__":
    main()
