"""
Visualization utilities for Harmony From First Principles.

This module provides plotting and visualization functions for:
- Weight heatmaps showing what notes each band member plays
- Spectrograms of generated audio
- Dissonance matrices between notes
- Loss curves over optimization
- Piano roll representations
- Pitch-class colored visualizations
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import io
import base64

from .band import BandMember, PolyphonicMember, MonophonicMember, DrumMember
from .tuning import TuningSystem, TwelveTET
from .dissonance import DissonanceCalculator
from .constraints import ConstraintSet, UserConstraint


# Pitch class names and colors
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Create a colormap for pitch classes using HSV
PITCH_CLASS_COLORS = plt.cm.hsv(np.linspace(0, 1, 12, endpoint=False))


def hz_to_midi_key(freq: float) -> int:
    """Convert frequency to nearest MIDI key number."""
    return int(round(12 * np.log2(freq / 440.0) + 69))


def midi_key_to_pitch_class(key: int) -> int:
    """Get pitch class (0-11) from MIDI key."""
    return key % 12


def midi_key_to_name(key: int) -> str:
    """Get note name (e.g., 'C4') from MIDI key."""
    octave = (key // 12) - 1
    pc = key % 12
    return f"{PITCH_CLASSES[pc]}{octave}"


def color_weights_by_pitch_class(weights: np.ndarray) -> np.ndarray:
    """Color weights by pitch class using HSV colormap.
    
    Args:
        weights: Array of shape (num_keys, num_beats) with weight values
        
    Returns:
        RGB array of shape (num_keys, num_beats, 3) with pitch-class coloring
    """
    num_keys, num_beats = weights.shape
    
    # Create color stick - each key gets a color based on pitch class
    color_stick = np.zeros((num_keys, 3))
    for key in range(num_keys):
        pc = key % 12
        color_stick[key] = PITCH_CLASS_COLORS[pc][:3]
    
    # Normalize weights to [0, 1] for visualization
    weight_max = weights.max()
    if weight_max > 0:
        weights_normalized = weights / weight_max
    else:
        weights_normalized = weights
    
    # Apply color: weights shape (K, B), color_stick shape (K, 3)
    # Result: (K, B, 3)
    colored = weights_normalized[..., np.newaxis] * color_stick[:, np.newaxis, :]
    
    return np.clip(colored, 0, 1)


def plot_weights(
    weights: Union[np.ndarray, torch.Tensor],
    title: str = "Weights",
    tuning: Optional[TuningSystem] = None,
    member: Optional[BandMember] = None,
    figsize: Tuple[int, int] = (10, 8),
    show_pitch_classes: bool = True
) -> Figure:
    """Plot weights as a heatmap with optional pitch-class coloring.
    
    Args:
        weights: Weight matrix [num_keys, num_beats]
        title: Plot title
        tuning: Tuning system (for labeling)
        member: Band member (for key range info)
        figsize: Figure size (width, height)
        show_pitch_classes: If True, color by pitch class
        
    Returns:
        Matplotlib Figure object
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    num_keys, num_beats = weights.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if show_pitch_classes:
        # Use pitch-class colored visualization
        colored = color_weights_by_pitch_class(weights)
        im = ax.imshow(colored, aspect='auto', interpolation='nearest')
        
        # Add color bar for pitch classes
        sm = plt.cm.ScalarMappable(
            cmap=LinearSegmentedColormap.from_list('pitch_classes', PITCH_CLASS_COLORS),
            norm=plt.Normalize(0, 11)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_ticks(range(12))
        cbar.set_ticklabels(PITCH_CLASSES)
        cbar.set_label('Pitch Class')
    else:
        # Standard heatmap
        im = ax.imshow(weights, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Weight')
    
    # Set labels
    ax.set_xlabel('Beat')
    ax.set_ylabel('Key')
    ax.set_title(title)
    
    # Set x-ticks
    ax.set_xticks(range(num_beats))
    ax.set_xticklabels([str(i) for i in range(num_beats)])
    
    # Set y-ticks - show C notes with octave labels
    c_keys = list(range(0, num_keys, 12))
    c_labels = []
    key_offset = member.key_offset if member else 0
    for k in c_keys:
        midi_key = key_offset + k
        octave = (midi_key // 12) - 1
        c_labels.append(f"{k}\n(C{octave})")
    ax.set_yticks(c_keys)
    ax.set_yticklabels(c_labels, fontsize=8)
    
    # Add grid for beats
    for i in range(num_beats):
        ax.axvline(x=i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 22050,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (12, 6),
    n_fft: int = 2048,
    hop_length: int = 512,
    color_by_pitch: bool = True
) -> Figure:
    """Plot spectrogram with optional pitch-class coloring.
    
    Args:
        audio: Audio array (1D)
        sample_rate: Sample rate in Hz
        title: Plot title
        figsize: Figure size
        n_fft: FFT size
        hop_length: Hop length for STFT
        color_by_pitch: If True, color by pitch class
        
    Returns:
        Matplotlib Figure object
    """
    import librosa
    
    # Compute spectrogram
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    num_bins, num_frames = S_db.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by_pitch:
        # Create pitch-class coloring for frequency bins
        color_stick = np.zeros((num_bins, 3))
        for bin_idx in range(num_bins):
            if bin_idx == 0:
                continue
            # Convert bin index to frequency
            freq = bin_idx * sample_rate / (2 * num_bins)
            midi_key = hz_to_midi_key(freq)
            pc = midi_key % 12
            color_stick[bin_idx] = PITCH_CLASS_COLORS[pc][:3]
        
        # Normalize spectrogram to [0, 1] for coloring
        S_min, S_max = S_db.min(), S_db.max()
        S_normalized = (S_db - S_min) / (S_max - S_min)
        
        # Apply color
        colored = S_normalized[..., np.newaxis] * color_stick[:, np.newaxis, :]
        colored = np.clip(colored, 0, 1)
        
        im = ax.imshow(colored, aspect='auto', origin='lower', interpolation='nearest')
    else:
        # Standard spectrogram
        img = librosa.display.specshow(
            S_db, sr=sample_rate, hop_length=hop_length,
            x_axis='time', y_axis='log', ax=ax, cmap='viridis'
        )
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    ax.set_title(title)
    ax.set_xlabel('Time (s)' if not color_by_pitch else 'Frame')
    ax.set_ylabel('Frequency (Hz)' if not color_by_pitch else 'Bin')
    
    if color_by_pitch:
        # Add frequency axis labels
        freq_ticks = [100, 200, 500, 1000, 2000, 5000, 10000]
        bin_ticks = [int(f * 2 * num_bins / sample_rate) for f in freq_ticks]
        valid_ticks = [(b, f) for b, f in zip(bin_ticks, freq_ticks) if 0 <= b < num_bins]
        if valid_ticks:
            bins, freqs = zip(*valid_ticks)
            ax.set_yticks(bins)
            ax.set_yticklabels([f"{f}" for f in freqs])
        ax.set_ylim(0, num_bins)
    
    plt.tight_layout()
    return fig


def plot_dissonance_matrix(
    D: Union[np.ndarray, torch.Tensor],
    tuning: Optional[TuningSystem] = None,
    title: str = "Dissonance Matrix",
    figsize: Tuple[int, int] = (10, 8),
    max_freq: Optional[int] = None
) -> Figure:
    """Plot dissonance matrix between notes.
    
    Args:
        D: Dissonance matrix [num_keys, num_keys]
        tuning: Tuning system
        title: Plot title
        figsize: Figure size
        max_freq: Maximum frequency to show (for limiting range)
        
    Returns:
        Matplotlib Figure object
    """
    if isinstance(D, torch.Tensor):
        D = D.detach().cpu().numpy()
    
    num_keys = D.shape[0]
    
    # Limit range if specified
    if max_freq is not None and tuning is not None:
        # Find keys within frequency range
        freqs = [tuning.get_frequency(k) for k in range(num_keys)]
        valid_keys = [k for k, f in enumerate(freqs) if f <= max_freq]
        if valid_keys:
            max_key = max(valid_keys)
            D = D[:max_key+1, :max_key+1]
            num_keys = max_key + 1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use log scale for better visualization (dissonance has large dynamic range)
    D_display = np.log1p(D)  # log(1 + x) to handle zeros
    
    im = ax.imshow(D_display, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Log Dissonance')
    
    ax.set_xlabel('Key')
    ax.set_ylabel('Key')
    ax.set_title(title)
    
    # Add pitch class labels if showing octave range
    if num_keys <= 128:
        # Show pitch class labels
        pc_labels = [PITCH_CLASSES[k % 12] for k in range(num_keys)]
        # Show every 12th key label
        tick_positions = list(range(0, num_keys, 12))
        tick_labels = [f"{k}\n({pc_labels[k]})" for k in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_loss_history(
    history: List[Dict[str, float]],
    title: str = "Loss History",
    figsize: Tuple[int, int] = (12, 6),
    log_scale: bool = False
) -> Figure:
    """Plot loss curves over optimization steps.
    
    Args:
        history: List of loss dictionaries, one per step
        title: Plot title
        figsize: Figure size
        log_scale: If True, use log scale for y-axis
        
    Returns:
        Matplotlib Figure object
    """
    if not history:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No loss history available",
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Extract loss components
    steps = list(range(len(history)))
    
    # Get all loss keys (excluding 'total')
    loss_keys = [k for k in history[0].keys() if k != 'total']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Total loss
    total_loss = [h.get('total', 0) for h in history]
    ax1.plot(steps, total_loss, linewidth=2, color='blue')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss')
    if log_scale and min(total_loss) > 0:
        ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: All components
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_keys)))
    for key, color in zip(loss_keys, colors):
        values = [h.get(key, 0) for h in history]
        ax2.plot(steps, values, label=key, color=color, linewidth=1.5)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss Value')
    ax2.set_title('Loss Components')
    ax2.legend(loc='upper right', fontsize=8)
    if log_scale:
        ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def create_weight_piano_roll(
    members: List[BandMember],
    constraints: Optional[ConstraintSet] = None,
    threshold: float = 0.1,
    title: str = "Piano Roll",
    figsize: Tuple[int, int] = (14, 8)
) -> Figure:
    """Create an interactive-style piano roll showing all members.
    
    Args:
        members: List of band members
        constraints: Optional constraints to overlay
        threshold: Weight threshold for showing notes
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    # Filter out drums for piano roll
    pitched_members = [m for m in members if not isinstance(m, DrumMember)]
    
    if not pitched_members:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No pitched members to display",
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Find global key range
    min_key = min(m.key_offset for m in pitched_members)
    max_key = max(m.key_offset + m.num_keys for m in pitched_members)
    max_beats = max(m.num_beats for m in pitched_members)
    
    # Create figure with subplots for each member
    fig, axes = plt.subplots(len(pitched_members), 1, figsize=figsize, sharex=True)
    
    if len(pitched_members) == 1:
        axes = [axes]
    
    # Color for each member
    member_colors = plt.cm.Set2(np.linspace(0, 1, len(pitched_members)))
    
    for idx, (member, ax) in enumerate(zip(pitched_members, axes)):
        weights = member.weights.detach().cpu().numpy()
        
        # Create binary mask of active notes
        active_mask = weights > threshold
        
        # Plot as imshow
        colored = color_weights_by_pitch_class(weights)
        
        # Use the colored visualization
        ax.imshow(colored, aspect='auto', interpolation='nearest', 
                 extent=[-0.5, member.num_beats - 0.5, 
                        member.key_offset, member.key_offset + member.num_keys],
                 origin='lower')
        
        # Overlay constraints if present
        if constraints:
            member_constraints = constraints.get_for_member(member.name)
            for constraint in member_constraints:
                beat = constraint.beat_index
                for key_idx in constraint.key_indices:
                    midi_key = member.key_offset + key_idx
                    # Draw a red box around constrained notes
                    rect = plt.Rectangle(
                        (beat - 0.5, midi_key - 0.5), 1, 1,
                        fill=False, edgecolor='red', linewidth=2
                    )
                    ax.add_patch(rect)
        
        ax.set_ylabel(f'{member.name}\n(Key)')
        ax.set_ylim(member.key_offset, member.key_offset + member.num_keys)
        
        # Add horizontal grid lines for octaves
        for k in range(member.key_offset, member.key_offset + member.num_keys, 12):
            ax.axhline(y=k, color='white', linewidth=0.5, alpha=0.3)
    
    # Common x-axis
    axes[-1].set_xlabel('Beat')
    axes[-1].set_xticks(range(max_beats))
    axes[-1].set_xticklabels([str(i) for i in range(max_beats)])
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def fig_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64 string for Gradio.
    
    Args:
        fig: Matplotlib Figure
        
    Returns:
        Base64-encoded PNG image string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64


def fig_to_numpy(fig: Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array for Gradio.
    
    Args:
        fig: Matplotlib Figure
        
    Returns:
        Numpy array representation of the figure
    """
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def save_audio(
    audio: np.ndarray,
    filename: str,
    sample_rate: int = 22050
) -> None:
    """Save audio as WAV file.
    
    Args:
        audio: Audio array (1D)
        filename: Output filename (should end in .wav)
        sample_rate: Sample rate in Hz
    """
    import scipy.io.wavfile as wavfile
    
    # Normalize to int16 range
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, audio_int16)


def save_weights_plot(
    members: List[BandMember],
    filename: str,
    constraints: Optional[ConstraintSet] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """Save comprehensive weights visualization as PNG.
    
    Creates a multi-panel figure showing:
    - Piano roll of all members
    - Individual weight heatmaps for each member
    
    Args:
        members: List of band members
        filename: Output filename (should end in .png)
        constraints: Optional constraints to overlay
        figsize: Figure size
    """
    pitched_members = [m for m in members if not isinstance(m, DrumMember)]
    
    if not pitched_members:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No pitched members to display",
                ha='center', va='center', transform=ax.transAxes)
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return
    
    # Create a grid layout
    n_members = len(pitched_members)
    n_cols = min(2, n_members)
    n_rows = (n_members + n_cols - 1) // n_cols + 1  # +1 for piano roll
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
    
    # Top: Piano roll spanning full width
    ax_piano = fig.add_subplot(gs[0, :])
    
    # Find global key range
    max_beats = max(m.num_beats for m in pitched_members)
    
    # Create piano roll visualization
    member_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for member_idx, member in enumerate(pitched_members):
        weights = member.weights.detach().cpu().numpy()
        
        # Plot active notes
        for beat in range(member.num_beats):
            active_keys = np.where(weights[:, beat] > 0.1)[0]
            for key_idx in active_keys:
                midi_key = member.key_offset + key_idx
                weight = weights[key_idx, beat]
                color = member_colors[member_idx % len(member_colors)]
                # Draw note as rectangle
                rect = plt.Rectangle(
                    (beat - 0.4, midi_key - 0.4), 0.8, 0.8,
                    facecolor=color, alpha=min(1.0, weight),
                    edgecolor='black', linewidth=0.5
                )
                ax_piano.add_patch(rect)
    
    # Add constraint indicators
    if constraints:
        for member in pitched_members:
            member_constraints = constraints.get_for_member(member.name)
            for constraint in member_constraints:
                beat = constraint.beat_index
                for key_idx in constraint.key_indices:
                    midi_key = member.key_offset + key_idx
                    rect = plt.Rectangle(
                        (beat - 0.4, midi_key - 0.4), 0.8, 0.8,
                        fill=False, edgecolor='red', linewidth=2, linestyle='--'
                    )
                    ax_piano.add_patch(rect)
    
    ax_piano.set_xlim(-0.5, max_beats - 0.5)
    ax_piano.set_ylim(
        min(m.key_offset for m in pitched_members) - 1,
        max(m.key_offset + m.num_keys for m in pitched_members) + 1
    )
    ax_piano.set_xlabel('Beat')
    ax_piano.set_ylabel('MIDI Key')
    ax_piano.set_title('Piano Roll - All Members')
    ax_piano.set_xticks(range(max_beats))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=member_colors[i % len(member_colors)], 
              label=m.name, alpha=0.7)
        for i, m in enumerate(pitched_members)
    ]
    if constraints and constraints.constraints:
        legend_elements.append(
            Patch(facecolor='none', edgecolor='red', linestyle='--',
                  label='Constraints', linewidth=2)
        )
    ax_piano.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Individual member plots
    for idx, member in enumerate(pitched_members):
        row = (idx // n_cols) + 1
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        weights = member.weights.detach().cpu().numpy()
        colored = color_weights_by_pitch_class(weights)
        
        ax.imshow(colored, aspect='auto', interpolation='nearest', origin='lower')
        ax.set_xlabel('Beat')
        ax.set_ylabel('Key Index')
        ax.set_title(f'{member.name} Weights')
        ax.set_xticks(range(member.num_beats))
    
    plt.suptitle('Harmony Optimization Results', fontsize=14, y=0.98)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_optimization_gif(
    history_frames: List[np.ndarray],
    filename: str,
    duration: float = 0.2
) -> None:
    """Create GIF from optimization history frames.
    
    Args:
        history_frames: List of weight matrices from optimization steps
        filename: Output filename (should end in .gif)
        duration: Duration per frame in seconds
    """
    try:
        from PIL import Image
        
        frames = []
        for weights in history_frames:
            fig = plot_weights(weights, title="Optimization Progress")
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            buf.seek(0)
            frames.append(Image.open(buf))
        
        # Save as GIF
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=int(duration * 1000),
            loop=0
        )
        plt.close('all')
    except ImportError:
        print("PIL not available, skipping GIF creation")


# ==================== VERIFICATION TESTS ====================

if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZATION UTILITIES VERIFICATION")
    print("=" * 60)
    
    # Create dummy data for testing
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test 1: color_weights_by_pitch_class
    print("\n1. Testing color_weights_by_pitch_class:")
    weights = np.random.rand(24, 4)  # 2 octaves, 4 beats
    colored = color_weights_by_pitch_class(weights)
    print(f"   Input shape: {weights.shape}")
    print(f"   Output shape: {colored.shape}")
    assert colored.shape == (24, 4, 3), "Colored weights shape mismatch!"
    print("   ✓ Shape correct")
    
    # Test 2: plot_weights
    print("\n2. Testing plot_weights:")
    weights_torch = torch.rand(24, 4)
    fig = plot_weights(weights_torch, title="Test Weights")
    print(f"   Figure created: {type(fig)}")
    plt.close(fig)
    print("   ✓ Figure created successfully")
    
    # Test 3: plot_loss_history
    print("\n3. Testing plot_loss_history:")
    history = [
        {'total': 10.0 - i*0.05, 'within': 5.0, 'temporal': 2.0, 'cross': 3.0}
        for i in range(100)
    ]
    fig = plot_loss_history(history, title="Test Loss")
    print(f"   Figure created with {len(history)} steps")
    plt.close(fig)
    print("   ✓ Loss history plotted")
    
    # Test 4: plot_dissonance_matrix
    print("\n4. Testing plot_dissonance_matrix:")
    D = np.random.rand(24, 24)
    D = (D + D.T) / 2  # Make symmetric
    fig = plot_dissonance_matrix(D, title="Test Dissonance")
    print(f"   Figure created for {D.shape} matrix")
    plt.close(fig)
    print("   ✓ Dissonance matrix plotted")
    
    # Test 5: fig_to_base64
    print("\n5. Testing fig_to_base64:")
    fig = plot_weights(np.random.rand(12, 4))
    b64 = fig_to_base64(fig)
    print(f"   Base64 string length: {len(b64)} characters")
    assert len(b64) > 0, "Empty base64 string!"
    print("   ✓ Base64 conversion successful")
    
    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)
