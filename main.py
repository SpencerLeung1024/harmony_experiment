import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.io import wavfile
import librosa
import sounddevice

from song import Song
from members import get_member
from audio_service import AudioService
from color_service import ColorService
from loss_handler import LossHandler
from optim_handler import OptimHandler

OUTPUT_FOLDER = "output"
SAMPLE_RATE = 22050
NUM_BINS = 4096

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

def plot_weights(song: Song, title_suffix: str = ""):
    num_members = len(song.members)
    cols = int(np.ceil(np.sqrt(num_members)))
    rows = int(np.ceil(num_members / cols))
    positions = [(i // cols, i % cols) for i in range(num_members)]

    # Always return a 2D array of axes even with 1 or 2 members so positions work
    fig, axs = plt.subplots(rows, cols, squeeze=False, figsize=(16, 10))
    for member_idx, (axx, axy) in enumerate(positions):
        member = song.members[member_idx]
        this_ax = axs[axx, axy]

        colormap = ColorService.color_weights(member)

        this_ax.imshow(colormap, aspect="auto")
        title = f"{member.name} Weights"
        if title_suffix:
            title += f" {title_suffix}"
        this_ax.set_title(title)
        this_ax.set_xlabel("Note Index")
        this_ax.set_ylabel("Key Index")
    
    plt.tight_layout()
    return fig

def plot_spectrogram(audio: np.ndarray, sample_rate: int, title_suffix: str = ""):
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

def plot_loss_history(loss_history: list):
    """Plot the loss components over optimization steps."""
    if not loss_history:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot total loss
    steps = [entry['step'] for entry in loss_history]
    total_losses = [entry['total'] for entry in loss_history]
    ax.plot(steps, total_losses, label='Total Loss', linewidth=2)
    
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Over Optimization Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

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
            tick_duration=tick_duration,
            total_ticks=total_ticks,
            ticks_per_note=24
        ),
        get_member(
            "guitar",
            tick_duration=tick_duration,
            total_ticks=total_ticks,
            ticks_per_note=2
        ),
        get_member(
            "bass",
            tick_duration=tick_duration,
            total_ticks=total_ticks,
            ticks_per_note=6
        ),
        # get_member(
        #     "synth",
        #     tick_duration=tick_duration,
        #     total_ticks=total_ticks,
        #     ticks_per_note=1
        # )
    ])

    # Create LossHandler and OptimHandler
    print("Initializing loss handler (computing dissonance matrices)...")
    song.loss_handler = LossHandler(song)
    
    print("Initializing optimizer...")
    song.optim_handler = OptimHandler(song, song.loss_handler)
    
    # Plot initial state
    print("\nRendering initial audio and plots...")
    audio = AudioService.render(song)
    save_audio(audio, f"{OUTPUT_FOLDER}/audio_initial.wav", SAMPLE_RATE)
    
    fig_weights = plot_weights(song, "(Initial)")
    fig_weights.savefig(f"{OUTPUT_FOLDER}/weights_initial.png")
    
    fig_spectrogram = plot_spectrogram(audio, SAMPLE_RATE, "(Initial)")
    fig_spectrogram.savefig(f"{OUTPUT_FOLDER}/spectrogram_initial.png")
    
    plt.show()
    
    # Optimization loop with user input
    loss_history = []
    
    while True:
        # Ask user for number of steps
        print(f"\nCurrent optimization step: {song.optim_handler.steps}")
        user_input = input("How many steps do you want to run? (0 to exit): ")
        
        try:
            num_steps = int(user_input)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        
        if num_steps == 0:
            print("Exiting optimization loop.")
            break
        
        if num_steps < 0:
            print("Please enter a positive integer.")
            continue
        
        # Run optimization steps
        print(f"Running {num_steps} optimization steps...")
        step_history = song.optim_handler.do_steps(num_steps)
        loss_history.extend(step_history)
        
        # Print loss summary
        latest_loss = step_history[-1]
        print(f"Step {latest_loss['step']}: Total Loss = {latest_loss['total']:.4f}")
        
        # Render and plot
        print("Rendering audio and plots...")
        audio = AudioService.render(song)
        
        # Save files with step number
        step_str = f"step{song.optim_handler.steps:04d}"
        save_audio(audio, f"{OUTPUT_FOLDER}/audio_{step_str}.wav", SAMPLE_RATE)
        
        fig_weights = plot_weights(song, f"({step_str})")
        fig_weights.savefig(f"{OUTPUT_FOLDER}/weights_{step_str}.png")
        
        fig_spectrogram = plot_spectrogram(audio, SAMPLE_RATE, f"({step_str})")
        fig_spectrogram.savefig(f"{OUTPUT_FOLDER}/spectrogram_{step_str}.png")
        
        # Plot loss history
        if len(loss_history) > 1:
            fig_loss = plot_loss_history(loss_history)
            fig_loss.savefig(f"{OUTPUT_FOLDER}/loss_history.png")
        
        # Play audio (do this right before plt.show() because generating colormaps takes a long time)
        sounddevice.play(audio, SAMPLE_RATE)
        
        plt.show()
    
    # Save final outputs
    print("\nSaving final outputs...")
    audio = AudioService.render(song)
    save_audio(audio, f"{OUTPUT_FOLDER}/audio_final.wav", SAMPLE_RATE)
    
    fig_weights = plot_weights(song, "(Final)")
    fig_weights.savefig(f"{OUTPUT_FOLDER}/weights_final.png")
    
    fig_spectrogram = plot_spectrogram(audio, SAMPLE_RATE, "(Final)")
    fig_spectrogram.savefig(f"{OUTPUT_FOLDER}/spectrogram_final.png")
    
    if loss_history:
        fig_loss = plot_loss_history(loss_history)
        fig_loss.savefig(f"{OUTPUT_FOLDER}/loss_history.png")
    
    plt.show()
    
    print(f"\nDone! All outputs saved to {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()
