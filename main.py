from typing import List
import numpy as np
import torch
import sys
import os
from matplotlib import pyplot as plt
from scipy.io import wavfile
import librosa
import sounddevice

from song import Song
from members import Member, PolyphonicMember, MonophonicMember, get_member
from instruments import Instrument, get_instrument
from tuning_systems import TuningSystem, get_tuning_system
from loss_handler import LossHandler
from optim_handler import OptimHandler
from audio_service import AudioService
from color_service import ColorService

OUTPUT_FOLDER = "output"
SAMPLE_RATE = 22050
NUM_BINS = 4096

# I was gonna os.rmdir but for safety leave that as a manual decision in VSCode
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def hz_to_bin(freq: float, num_bins: int, sample_rate: int) -> int:
    return int((freq / (sample_rate / 2)) * num_bins)

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_name_to_midi(note_name: str) -> int:
    pitch_class = None
    octave = None
    # Handle octave -1
    if note_name[-2] == '-':
        pitch_class = note_name[:-2]
        octave = -1
    else:
        pitch_class = note_name[:-1]
        octave = int(note_name[-1])
    
    pitch_class_index = PITCH_CLASSES.index(pitch_class)
    midi_number = (octave + 1) * 12 + pitch_class_index
    return midi_number

# Awful function I made up in a few minutes
# In the real implementation the user would paint in weights on the Gradio UI
def note_names_to_weightsmap(member: Member, velocity: float, note_names: str) -> torch.Tensor:
    instrument_range_low = member.instrument_range[0]
    weightsmap = torch.zeros_like(member.weights)
    lines = note_names.strip().split('\n')
    for note_idx, line in enumerate(lines):
        chord = line.strip()
        note_names = chord.split(' ')
        for note_name in note_names:
            midi_number = note_name_to_midi(note_name)
            key_idx = midi_number - instrument_range_low
            if 0 <= key_idx < weightsmap.shape[0]:
                weightsmap[key_idx, note_idx] = velocity
    return weightsmap

# from v2\harmony\visualization.py
def save_audio(
    audio: np.ndarray,
    filename: str,
    sample_rate: int,
):
    # Normalize to int16 range
    '''
    audio_int = (audio * 32767).astype(np.int32)
    samples_clipping = np.sum(np.abs(audio_int) > 32767)
    if samples_clipping > 0:
        print(f"Warning: {samples_clipping} out of {len(audio_int)} samples are clipping.")
        audio_int = np.clip(audio_int, -32767, 32767)
    audio_int16 = audio_int.astype(np.int16)
    '''
    # Now that AudioService.apply_limiter is used there should be no more clipping
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, audio_int16)

def plot_activations(song: Song, title_suffix: str = "", interactive: bool = True):
    num_members = len(song.members)
    cols = int(np.ceil(np.sqrt(num_members)))
    rows = int(np.ceil(num_members / cols))
    positions = [(i // cols, i % cols) for i in range(num_members)]

    # Always return a 2D array of axes even with 1 or 2 members so positions work
    fig, axs = plt.subplots(rows, cols, squeeze=False, figsize=(16, 10))
    for member_idx, (axx, axy) in enumerate(positions):
        member = song.members[member_idx]
        this_ax = axs[axx, axy]

        colormap = ColorService.color_activations(member)

        this_ax.imshow(colormap, aspect="auto")
        title = f"{member.name} Activations"
        if title_suffix:
            title += f" {title_suffix}"
        this_ax.set_title(title)
        this_ax.set_xlabel("Note Index")
        this_ax.set_ylabel("Key Index")
    
    plt.tight_layout()
    return fig

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

def plot_loss_history(loss_history: list, interactive: bool = True):
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

# Refactored out of the main function
def output_results(song: Song, step_str: str, interactive: bool):
    # Try to get the actual integer step, or None if step_str is "initial" or "final"
    step = None
    if step_str.startswith("step"):
        try:
            step = int(step_str[4:])
        except ValueError:
            pass
    
    print(f"Rendering audio and plots for {step_str}...")

    audio = AudioService.render(song)
    audio = AudioService.apply_limiter(audio)
    save_audio(audio, f"{OUTPUT_FOLDER}/audio_{step_str}.wav", song.sample_rate)
    
    fig_activations = plot_activations(song, f"({step_str})", interactive)
    fig_activations.savefig(f"{OUTPUT_FOLDER}/activations_{step_str}.png")
    
    fig_spectrogram = plot_spectrogram(audio, song.sample_rate, f"({step_str})", interactive)
    fig_spectrogram.savefig(f"{OUTPUT_FOLDER}/spectrogram_{step_str}.png")
    
    if interactive:
        if step is not None:
            sounddevice.play(audio, song.sample_rate)
        plt.show()

def main(step_list: List[str]):
    # Are we in interactive or straight through mode?
    interactive = len(step_list) == 0

    if not interactive:
        # Convert step_list to integers
        for i in range(len(step_list)):
            try:
                step_list[i] = int(step_list[i])
            except ValueError:
                print(f"Invalid step value: {step_list[i]}. Please enter integers for steps.")
                return
        
        # Convert step_list from the step at which to stop, to the number of steps to advance
        if len(step_list) > 1:
            for i in range(len(step_list) - 1, 0, -1):
                step_list[i] = step_list[i] - step_list[i-1]

    # Create a song
    song = Song(
        measures=8,
        tempo=120,
        beats_per_measure=4,
        ticks_per_beat=2,
        sample_rate=SAMPLE_RATE
    )

    # Create members
    tick_duration = song.tick_duration()
    total_ticks = song.total_ticks()

    # Use other tuning systems
    # test_tuning_system = get_tuning_system("Non-Octave System", step_ratio=2.0**(1/12), divisions=1, reference_freq=440.0)
    # print(f"{test_tuning_system.keys.shape[0]} keys")

    # test_instrument = PolyphonicMember(
    #     name="piano",
    #     instrument=get_instrument("piano"),
    #     tuning_system=test_tuning_system,
    #     instrument_range=[0,test_tuning_system.keys.shape[0]-1],
    #     velocity=0.5,
    #     tick_duration=tick_duration,
    #     total_ticks=total_ticks,
    #     ticks_per_note=1,
    #     # hp
    #     # initial_weights
    # )

    song.members.extend([
        # test_instrument,
        
        get_member(
            "piano",
            velocity=0.5,
            tick_duration=tick_duration,
            total_ticks=total_ticks,
            ticks_per_note=6
        ),
        get_member(
            "guitar",
            velocity=0.2,
            tick_duration=tick_duration,
            total_ticks=total_ticks,
            ticks_per_note=1
        ),
        get_member(
            "bass",
            velocity=0.7, # Bring out the bass more my audio system is terrible
            tick_duration=tick_duration,
            total_ticks=total_ticks,
            ticks_per_note=3
        ),
    ])

    # Apply overrides to the piano
    song.members[0].hp = { # If mate dissonance losses are above 0.2 the piano hides away in the 7th octave where the guitar and bass cannot touch it
        'mate_concurrent': 0.0,
        'mate_temporal': 0.0,
        'extreme_range': 3.0 # Bonus to piano since it has a very wide range. The optimizer can settle on chords spanning multiple octaves
    }
#     weightsmap = note_names_to_weightsmap(song.members[0], 0.125, '''D3 F4 A3 D4
# A2 A4 C#3 E4
# D3 A4 D4 F4
# C3 C5 E4 G4
# F3 C5 F4 A4
# E3 G#4 B3 B4
# A2 A4 E4 C5
# G2 B4 G4 D5
# C3 C5 G4 D#5
# A#3 C5 G4 E5
# A3 C5 F4 F5
# C4 D#5 A4 F#5
# B3 D5 G4 G5
# A#3 F5 D5 G#5
# A3 F5 D5 A5
# A3 E5 C#5 A4''')
#     song.members[0].paint_weights(weightsmap)
    # When re-enabling, remember to make the piano have 16 notes throughout the song

    # Guitar
    song.members[1].hp = {
        'mate_concurrent': 3.0, # Stronger following of whatever chord the piano is playing
    }

    # Bass
    song.members[2].hp = {
        'mate_concurrent': 3.0, # Is that supposed to be funny?
    }
    # Was it not?
    # Sometimes the things you come out with are kind of bizarre!
    # Bizarre...

    # Create LossHandler and OptimHandler
    print("Initializing loss handler (computing dissonance matrices)...")
    song.loss_handler = LossHandler(song)
    
    print("Initializing optimizer...")
    song.optim_handler = OptimHandler(song, song.loss_handler)
    
    # Plot initial state
    output_results(song, "initial", interactive)
    
    # Optimization loop with user input
    loss_history = []
    
    while True:
        if interactive:
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
        else:
            # In non-interactive mode, just run through the specified steps
            if len(step_list) == 0:
                break
            num_steps = step_list.pop(0)
        
        # Run optimization steps
        print(f"Running {num_steps} optimization steps...")
        step_history = song.optim_handler.do_steps(num_steps)
        loss_history.extend(step_history)
        
        # Print loss summary
        latest_loss = step_history[-1]
        print(f"Step {latest_loss['step']}: Total Loss = {latest_loss['total']:.4f}")

        # Plot this step
        step_str = f"step{song.optim_handler.steps:04d}"
        output_results(song, step_str, interactive)
    
    # Plot final state
    output_results(song, "final", interactive)
    
    print(f"\nDone! All outputs saved to {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        print("Usage: python main.py [steps at which to save outputs, leave empty for interactive mode]")
        print("Example: python main.py 40 60 80 100")
        print("Will save outputs at initial, steps 40, 60, 80, and final at 100 steps.")
    else:
        step_list = sys.argv[1:] # Ignore the script name
        main(step_list)
