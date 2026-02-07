"""
Harmony From First Principles - Command-Line Demo

A command-line demonstration of the harmony optimization system.
No Gradio dependency - suitable for headless environments.

Examples:
    python demo.py --members piano guitar --tuning 12-TET --steps 100
    python demo.py --members piano --tuning 19-EDO --steps 200 --output-dir ./results
    python demo.py --members piano guitar bass drums --tuning pythagorean --duration 6
"""

import os
import sys
import argparse
import time
from typing import List, Optional

import numpy as np
import torch

from harmony.tuning import (
    TwelveTET, PythagoreanTuning, MeantoneTuning, 
    EDOSystem, NonOctaveSystem, TuningSystem
)
from harmony.band import PolyphonicMember, MonophonicMember, DrumMember, BandMember
from harmony.constraints import ConstraintSet, UserConstraint
from harmony.optimizer import HarmonyOptimizer
from harmony.visualization import (
    plot_weights, plot_loss_history, plot_spectrogram,
    create_weight_piano_roll, save_audio, save_weights_plot
)


# Available tuning systems
TUNING_SYSTEMS = {
    "12-tet": TwelveTET,
    "pythagorean": PythagoreanTuning,
    "quarter-meantone": lambda: MeantoneTuning(comma_fraction=0.25),
    "third-meantone": lambda: MeantoneTuning(comma_fraction=1/3),
    "19-edo": lambda: EDOSystem(divisions=19),
    "24-edo": lambda: EDOSystem(divisions=24),
    "31-edo": lambda: EDOSystem(divisions=31),
    "41-edo": lambda: EDOSystem(divisions=41),
    "53-edo": lambda: EDOSystem(divisions=53),
    "alpha": NonOctaveSystem.alpha_scale,
    "beta": NonOctaveSystem.beta_scale,
    "bohlen-pierce": NonOctaveSystem.bohlen_pierce,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Harmony From First Principles - Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--members", "-m",
        nargs="+",
        choices=["piano", "guitar", "bass", "drums"],
        default=["piano"],
        help="Band members to include"
    )
    
    parser.add_argument(
        "--tuning", "-t",
        type=str,
        choices=list(TUNING_SYSTEMS.keys()),
        default="12-tet",
        help="Tuning system to use"
    )
    
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=100,
        help="Number of optimization steps"
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=0.02,
        help="Learning rate for optimization"
    )
    
    parser.add_argument(
        "--density", "-d",
        type=float,
        default=0.15,
        help="Target note density (0-1)"
    )
    
    parser.add_argument(
        "--beats", "-b",
        type=int,
        default=4,
        help="Number of beats in composition"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Audio duration in seconds"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        choices=[22050, 44100],
        help="Audio sample rate"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Output directory for generated files"
    )
    
    parser.add_argument(
        "--constraints", "-c",
        nargs="+",
        default=None,
        help="Add constraints as 'member:beat:note' (e.g., 'piano:0:C4')"
    )
    
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio after generation (requires sounddevice)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def create_members(
    member_names: List[str],
    tuning: TuningSystem,
    num_beats: int
) -> List[BandMember]:
    """Create band members based on names."""
    members = []
    
    for name in member_names:
        if name == "piano":
            members.append(PolyphonicMember.piano(num_beats=num_beats, tuning=tuning))
        elif name == "guitar":
            members.append(MonophonicMember.guitar(num_beats=num_beats, tuning=tuning))
        elif name == "bass":
            members.append(MonophonicMember.bass(num_beats=num_beats, tuning=tuning))
        elif name == "drums":
            members.append(DrumMember.standard_rock(num_beats=num_beats, enabled=True))
    
    return members


def parse_constraints(
    constraint_strings: Optional[List[str]],
    members: List[BandMember]
) -> ConstraintSet:
    """Parse constraint strings into ConstraintSet."""
    constraint_set = ConstraintSet()
    
    if not constraint_strings:
        return constraint_set
    
    # Build member name lookup
    member_lookup = {m.name: m for m in members}
    
    for cs in constraint_strings:
        try:
            parts = cs.split(":")
            if len(parts) != 3:
                print(f"Warning: Invalid constraint format '{cs}', skipping")
                continue
            
            member_name, beat_str, note_name = parts
            
            if member_name not in member_lookup:
                print(f"Warning: Unknown member '{member_name}', skipping")
                continue
            
            member = member_lookup[member_name]
            beat = int(beat_str)
            
            # Parse note name to MIDI key
            note_name = note_name.strip().upper()
            pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            
            # Extract pitch class and octave
            if len(note_name) == 2:
                pc = note_name[0]
                octave = int(note_name[1])
            elif len(note_name) == 3 and note_name[1] == "#":
                pc = note_name[:2]
                octave = int(note_name[2])
            else:
                print(f"Warning: Invalid note format '{note_name}', skipping")
                continue
            
            pc_idx = pc_names.index(pc)
            midi_key = (octave + 1) * 12 + pc_idx
            
            # Convert to member key index
            key_idx = midi_key - member.key_offset
            
            if 0 <= key_idx < member.num_keys and 0 <= beat < member.num_beats:
                constraint = UserConstraint(
                    member_name=member_name,
                    beat_index=beat,
                    key_indices=key_idx,
                    strengths=1.0
                )
                constraint_set.add_constraint(constraint)
                print(f"  Added constraint: {member_name}, beat {beat}, note {note_name} (key {key_idx})")
            else:
                print(f"Warning: Constraint out of range '{cs}', skipping")
        
        except Exception as e:
            print(f"Warning: Failed to parse constraint '{cs}': {e}")
    
    return constraint_set


def note_to_name(midi_key: int) -> str:
    """Convert MIDI key to note name."""
    pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pc = midi_key % 12
    octave = (midi_key // 12) - 1
    return f"{pc_names[pc]}{octave}"


def print_results(optimizer: HarmonyOptimizer, members: List[BandMember], verbose: bool = False):
    """Print optimization results."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    for member in members:
        if isinstance(member, DrumMember):
            print(f"\n🥁 {member.name.upper()}:")
            print(f"   Pattern: {member.pattern}")
            print(f"   Enabled: {member.enabled}")
            continue
        
        print(f"\n🎹 {member.name.upper()}:")
        
        weights = member.weights.detach().numpy()
        
        for beat in range(member.num_beats):
            active = np.where(weights[:, beat] > 0.1)[0]
            if len(active) > 0:
                notes = []
                for key_idx in active[:5]:  # Limit to first 5 notes
                    midi_key = member.key_offset + key_idx
                    note_name = note_to_name(midi_key)
                    weight = weights[key_idx, beat]
                    notes.append(f"{note_name}({weight:.2f})")
                
                if len(active) > 5:
                    notes.append(f"... (+{len(active) - 5} more)")
                
                print(f"   Beat {beat}: {' | '.join(notes)}")
            else:
                print(f"   Beat {beat}: (rest)")


def main():
    """Run the demo."""
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("🎵 HARMONY FROM FIRST PRINCIPLES - DEMO")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Members: {', '.join(args.members)}")
    print(f"  Tuning: {args.tuning}")
    print(f"  Steps: {args.steps}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Target Density: {args.density}")
    print(f"  Beats: {args.beats}")
    print(f"  Duration: {args.duration}s")
    print(f"  Sample Rate: {args.sample_rate}Hz")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"  Output Directory: {args.output_dir}")
    
    # Create tuning system
    print(f"\nInitializing {args.tuning} tuning system...")
    tuning_class = TUNING_SYSTEMS[args.tuning]
    tuning = tuning_class()
    print(f"  Tuning: {tuning.name}")
    
    # Create band members
    print(f"\nCreating band members...")
    members = create_members(args.members, tuning, args.beats)
    for m in members:
        print(f"  ✓ {m.name}: {m.num_keys} keys, {m.num_beats} beats")
    
    # Parse constraints
    print(f"\nParsing constraints...")
    constraints = parse_constraints(args.constraints, members)
    if constraints.constraints:
        print(f"  Total constraints: {len(constraints.constraints)}")
    else:
        print("  No constraints specified")
    
    # Create optimizer
    print(f"\nInitializing optimizer...")
    loss_weights = {
        'within': 1.0,
        'cross': 1.0,
        'temporal': 0.5,
        'density': 10.0,
        'sparsity': 1.0,
        'range': 1.0,
        'interval_jump': 0.5
    }
    
    optimizer = HarmonyOptimizer(
        members=members,
        tuning=tuning,
        constraints=constraints,
        loss_weights=loss_weights,
        lr=args.learning_rate,
        target_density=args.density,
        temporal_decay=0.3
    )
    print("  ✓ Optimizer initialized")
    
    # Precompute dissonance
    print(f"\nPrecomputing dissonance matrices...")
    start_time = time.time()
    optimizer.precompute_dissonance()
    print(f"  ✓ Dissonance matrices computed in {time.time() - start_time:.2f}s")
    
    # Run optimization
    print(f"\nRunning optimization ({args.steps} steps)...")
    start_time = time.time()
    
    def progress_callback(step: int, loss_dict: dict):
        if args.verbose and step % 25 == 0:
            print(f"  Step {step + 1}/{args.steps}: Loss = {loss_dict['total']:.4f}")
    
    result = optimizer.optimize(
        steps=args.steps,
        callback=progress_callback,
        verbose=False
    )
    
    elapsed = time.time() - start_time
    print(f"  ✓ Optimization complete in {elapsed:.2f}s")
    print(f"  Final Loss: {result['final_loss']:.4f}")
    
    # Print results
    print_results(optimizer, members, verbose=args.verbose)
    
    # Generate audio
    print(f"\nGenerating audio...")
    audio = optimizer.get_audio(
        duration=args.duration,
        sample_rate=args.sample_rate
    )
    print(f"  Audio shape: {audio.shape}")
    print(f"  Duration: {len(audio) / args.sample_rate:.2f}s")
    
    # Save audio
    audio_path = os.path.join(args.output_dir, "generated_audio.wav")
    save_audio(audio, audio_path, args.sample_rate)
    print(f"  ✓ Audio saved to: {audio_path}")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    # Piano roll
    viz_path = os.path.join(args.output_dir, "visualization.png")
    save_weights_plot(members, viz_path, constraints)
    print(f"  ✓ Visualization saved to: {viz_path}")
    
    # Loss history
    if optimizer.loss_history:
        loss_fig = plot_loss_history(optimizer.loss_history, title="Loss History")
        loss_path = os.path.join(args.output_dir, "loss_history.png")
        loss_fig.savefig(loss_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Loss history saved to: {loss_path}")
    
    # Individual member plots
    for member in members:
        if isinstance(member, DrumMember):
            continue
        
        fig = plot_weights(
            member.weights,
            title=f"{member.name} Weights ({tuning.name})",
            member=member
        )
        weight_path = os.path.join(args.output_dir, f"{member.name}_weights.png")
        fig.savefig(weight_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ {member.name} weights saved to: {weight_path}")
    
    # Spectrogram
    try:
        import librosa
        spec_fig = plot_spectrogram(audio, args.sample_rate, title="Spectrogram")
        spec_path = os.path.join(args.output_dir, "spectrogram.png")
        spec_fig.savefig(spec_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Spectrogram saved to: {spec_path}")
    except Exception as e:
        print(f"  ⚠ Spectrogram creation failed: {e}")
    
    # Play audio if requested
    if args.play:
        print(f"\nPlaying audio...")
        try:
            import sounddevice as sd
            sd.play(audio, args.sample_rate)
            sd.wait()
            print("  ✓ Playback complete")
        except ImportError:
            print("  ⚠ sounddevice not available, skipping playback")
        except Exception as e:
            print(f"  ⚠ Playback failed: {e}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nOutput files in: {os.path.abspath(args.output_dir)}")
    print("\nNext steps:")
    print("  - Listen to generated_audio.wav")
    print("  - View visualization.png for piano roll")
    print("  - Check loss_history.png for optimization progress")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
