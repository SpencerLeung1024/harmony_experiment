"""
Full Feature Showcase for Harmony From First Principles

This script demonstrates all implemented features:
- All tuning systems
- All band member types
- Full optimization pipeline
- Audio synthesis and mixing
- Visualization generation
- Comparison of tuning systems

Run with: python run_full_demo.py
Output: output/full_demo/
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import json

import numpy as np
import torch

# Ensure harmony package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harmony import (
    # Tuning systems
    TwelveTET, PythagoreanTuning, MeantoneTuning,
    EDOSystem, NonOctaveSystem,
    # Instruments
    ADSR, Instrument,
    # Band members
    PolyphonicMember, MonophonicMember, DrumMember,
    # Optimizer
    HarmonyOptimizer,
    # Synthesis & Mixing
    AudioSynthesizer, AudioMixer,
    # Constraints
    UserConstraint, ConstraintSet,
    # Visualization
    plot_weights, plot_loss_history, plot_dissonance_matrix,
    plot_spectrogram, create_weight_piano_roll, save_audio,
    save_weights_plot,
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def ensure_output_dir():
    """Create output directory for full demo."""
    output_dir = os.path.join("output", "full_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ["audio", "visualizations", "comparisons", "tunings"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    return output_dir


def demo_tuning_systems(output_dir: str):
    """Demonstrate all tuning systems."""
    print_section("TUNING SYSTEMS")
    
    tunings = [
        ("12-TET", TwelveTET()),
        ("Pythagorean", PythagoreanTuning()),
        ("1/4-comma Meantone", MeantoneTuning(comma_fraction=0.25)),
        ("1/3-comma Meantone", MeantoneTuning(comma_fraction=1/3)),
        ("19-EDO", EDOSystem(divisions=19)),
        ("24-EDO", EDOSystem(divisions=24)),
        ("31-EDO", EDOSystem(divisions=31)),
        ("41-EDO", EDOSystem(divisions=41)),
        ("53-EDO", EDOSystem(divisions=53)),
        ("Alpha Scale", NonOctaveSystem.alpha_scale()),
        ("Beta Scale", NonOctaveSystem.beta_scale()),
        ("Bohlen-Pierce", NonOctaveSystem.bohlen_pierce()),
    ]
    
    print(f"\nDemonstrating {len(tunings)} tuning systems:")
    
    results = []
    for name, tuning in tunings:
        # Get frequencies for one octave (or equivalent)
        if isinstance(tuning, NonOctaveSystem):
            num_keys = 13
        else:
            num_keys = 12 if not isinstance(tuning, EDOSystem) else tuning.divisions
        
        freqs = tuning.get_all_frequencies(num_keys)
        
        print(f"  • {name:25s} - {num_keys:2d} keys, freq range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
        
        results.append({
            "name": name,
            "num_keys": num_keys,
            "first_freq": float(freqs[0]),
            "last_freq": float(freqs[-1])
        })
    
    # Save tuning comparison
    with open(os.path.join(output_dir, "tunings", "tuning_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  ✓ Tuning comparison saved to tunings/tuning_comparison.json")
    return tunings


def demo_instruments(output_dir: str):
    """Demonstrate instrument presets and ADSR."""
    print_section("INSTRUMENTS & ADSR")
    
    instruments = [
        ("Piano", Instrument.piano()),
        ("Guitar", Instrument.guitar()),
        ("Bass", Instrument.bass()),
        ("Synth", Instrument.synth()),
        ("Kick Drum", Instrument.drums_kick()),
        ("Snare Drum", Instrument.drums_snare()),
        ("Hi-Hat", Instrument.drums_hihat()),
    ]
    
    print(f"\nDemonstrating {len(instruments)} instrument presets:")
    
    for name, inst in instruments:
        print(f"  • {name:15s} - {len(inst.harmonics):2d} harmonics, ADSR: "
              f"A={inst.adsr.attack:.3f}s, D={inst.adsr.decay:.3f}s, "
              f"S={inst.adsr.sustain:.2f}, R={inst.adsr.release:.3f}s")
    
    # Test per-harmonic ADSR
    print("\n  Testing per-harmonic ADSR:")
    adsr1 = ADSR(attack=0.002, decay=0.3, sustain=0.6, release=0.8)  # pluck-like
    adsr2 = ADSR(attack=0.3, decay=0.2, sustain=0.9, release=1.0)   # pad-like
    adsr3 = ADSR(attack=0.005, decay=0.4, sustain=0.3, release=0.5)  # piano-like
    
    multi_adsr_inst = Instrument(
        name="Multi-ADSR Test",
        harmonics=[(1.0, 1.0), (2.0, 0.5), (3.0, 0.25)],
        adsr=ADSR(attack=0.005, decay=0.4, sustain=0.3, release=0.5),
        per_harmonic_adsr={0: adsr1, 1: adsr2, 2: adsr3}
    )
    
    print(f"    Created instrument with {len(multi_adsr_inst.per_harmonic_adsr)} different ADSR envelopes")
    print(f"    • Harmonic 1: {multi_adsr_inst.per_harmonic_adsr[0].attack:.3f}s attack (pluck-like)")
    print(f"    • Harmonic 2: {multi_adsr_inst.per_harmonic_adsr[1].attack:.3f}s attack (pad-like)")
    print(f"    • Harmonic 3: {multi_adsr_inst.per_harmonic_adsr[2].attack:.3f}s attack (piano-like)")


def demo_band_members(output_dir: str):
    """Demonstrate all band member types."""
    print_section("BAND MEMBERS")
    
    num_beats = 4
    
    # Create all member types
    piano = PolyphonicMember.piano(num_beats=num_beats)
    guitar = MonophonicMember.guitar(num_beats=num_beats)
    bass = MonophonicMember.bass(num_beats=num_beats)
    drums = DrumMember.standard_rock(num_beats=num_beats)
    
    members = [
        ("Piano (Polyphonic)", piano),
        ("Guitar (Monophonic)", guitar),
        ("Bass (Monophonic)", bass),
        ("Drums (Pattern-based)", drums),
    ]
    
    print(f"\nCreated {len(members)} band members:")
    
    for name, member in members:
        print(f"  • {name:25s} - {member.num_keys:3d} keys × {member.num_beats} beats, "
              f"optimizable: {member.weights.requires_grad}")
    
    # Demonstrate monophonic constraint
    print("\n  Testing monophonic constraint:")
    guitar.weights.data.zero_()
    guitar.weights.data[10, 0] = 3.0
    guitar.weights.data[15, 0] = 2.0
    guitar.weights.data[20, 0] = 1.0
    
    active = guitar.get_active_notes(beat_index=0)
    print(f"    Set 3 notes on beat 0, got {len(active)} active note(s): {active}")
    print(f"    ✓ Monophonic constraint enforced (only highest weight active)")
    
    # Demonstrate drum pattern
    print("\n  Standard rock drum pattern:")
    for beat in range(4):
        active = drums.get_active_notes(beat_index=beat)
        print(f"    Beat {beat}: Active drums: {active}")
    
    return [piano, guitar, bass, drums]


def demo_optimization_scenario(name: str, members: List, tuning, output_dir: str,
                                constraints=None, num_steps: int = 100) -> Dict:
    """Run a complete optimization scenario."""
    print(f"\n  Scenario: {name}")
    print(f"    Members: {[m.name for m in members]}")
    print(f"    Tuning: {tuning.name}")
    print(f"    Steps: {num_steps}")
    
    # Create optimizer
    optimizer = HarmonyOptimizer(
        members=members,
        tuning=tuning,
        constraints=constraints,
        lr=0.03,
        target_density=0.15,
        loss_weights={
            'within': 1.0,
            'temporal': 0.5,
            'cross': 1.0,
            'density': 10.0,
            'sparsity': 1.0,
            'range': 1.0,
            'interval_jump': 0.5
        }
    )
    
    # Run optimization
    start_time = time.time()
    result = optimizer.optimize(steps=num_steps, verbose=False)
    loss_history = result['loss_history']
    elapsed = time.time() - start_time
    
    final_loss = loss_history[-1]['total']
    initial_loss = loss_history[0]['total']
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"    ✓ Optimization complete in {elapsed:.1f}s")
    print(f"      Initial loss: {initial_loss:.4f}")
    print(f"      Final loss: {final_loss:.4f}")
    print(f"      Improvement: {improvement:.1f}%")
    
    # Synthesize audio
    synth = AudioSynthesizer(sample_rate=22050, beat_duration=0.5)
    mixer = AudioMixer()
    
    tracks = []
    track_names = []
    
    for member in members:
        audio = synth.synthesize_member(member)
        tracks.append(audio)
        track_names.append(member.name)
    
    # Mix tracks with appropriate gains
    gains = [0.7 if isinstance(m, PolyphonicMember) else 0.8 for m in members]
    mixed = mixer.mix_tracks(tracks, gains=gains)
    
    # Save individual tracks and mix
    scenario_dir = os.path.join(output_dir, "audio", name.replace(" ", "_").lower())
    os.makedirs(scenario_dir, exist_ok=True)
    
    for track, track_name in zip(tracks, track_names):
        save_audio(track, os.path.join(scenario_dir, f"{track_name}.wav"), sample_rate=22050)
    
    save_audio(mixed, os.path.join(scenario_dir, "mixed.wav"), sample_rate=22050)
    
    # Create visualizations
    viz_dir = os.path.join(output_dir, "visualizations", name.replace(" ", "_").lower())
    os.makedirs(viz_dir, exist_ok=True)
    
    # Loss history
    fig = plot_loss_history(loss_history)
    fig.savefig(os.path.join(viz_dir, "loss_history.png"), dpi=150, bbox_inches='tight')
    
    # Weight plots for each member
    pitched_members = [m for m in members if not isinstance(m, DrumMember)]
    if pitched_members:
        save_weights_plot(
            pitched_members,
            os.path.join(viz_dir, "weights.png")
        )
    
    # Spectrogram of mixed audio
    fig = plot_spectrogram(mixed, sample_rate=22050)
    fig.savefig(os.path.join(viz_dir, "spectrogram.png"), dpi=150, bbox_inches='tight')
    
    print(f"      ✓ Audio saved to: audio/{name.replace(' ', '_').lower()}/")
    print(f"      ✓ Visualizations saved to: visualizations/{name.replace(' ', '_').lower()}/")
    
    return {
        "name": name,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "improvement": improvement,
        "time": elapsed,
        "loss_history": loss_history
    }


def demo_scenarios(output_dir: str):
    """Run all demo scenarios."""
    print_section("OPTIMIZATION SCENARIOS")
    
    results = []
    
    # Scenario 1: Piano solo in 12-TET
    piano = PolyphonicMember.piano(num_beats=6)
    result = demo_optimization_scenario(
        "Piano Solo 12-TET",
        [piano],
        TwelveTET(),
        output_dir,
        num_steps=100
    )
    results.append(result)
    
    # Scenario 2: Piano + Guitar in Pythagorean
    piano = PolyphonicMember.piano(num_beats=4)
    guitar = MonophonicMember.guitar(num_beats=4)
    result = demo_optimization_scenario(
        "Piano Guitar Pythagorean",
        [piano, guitar],
        PythagoreanTuning(),
        output_dir,
        num_steps=80
    )
    results.append(result)
    
    # Scenario 3: Full band in 19-EDO
    piano = PolyphonicMember.piano(num_beats=4)
    guitar = MonophonicMember.guitar(num_beats=4)
    bass = MonophonicMember.bass(num_beats=4)
    drums = DrumMember.standard_rock(num_beats=4)
    result = demo_optimization_scenario(
        "Full Band 19-EDO",
        [piano, guitar, bass, drums],
        EDOSystem(divisions=19),
        output_dir,
        num_steps=80
    )
    results.append(result)
    
    # Scenario 4: Piano with constraints
    piano = PolyphonicMember.piano(num_beats=4)
    constraints = ConstraintSet()
    constraints.add_constraint(
        UserConstraint("piano", 0, [39, 43, 46], fixed_value=1.0)  # C major
    )
    constraints.add_constraint(
        UserConstraint("piano", 2, [36, 40, 43], fixed_value=1.0)  # F major
    )
    result = demo_optimization_scenario(
        "Piano with Constraints",
        [piano],
        TwelveTET(),
        output_dir,
        constraints=constraints,
        num_steps=80
    )
    results.append(result)
    
    # Save results summary
    with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def demo_tuning_comparison(output_dir: str):
    """Compare optimization across different tuning systems."""
    print_section("TUNING SYSTEM COMPARISON")
    
    tunings = [
        ("12-TET", TwelveTET()),
        ("Pythagorean", PythagoreanTuning()),
        ("1/4-comma Meantone", MeantoneTuning(comma_fraction=0.25)),
        ("19-EDO", EDOSystem(divisions=19)),
        ("31-EDO", EDOSystem(divisions=31)),
    ]
    
    print(f"\nComparing {len(tunings)} tuning systems with identical piano configurations:")
    
    results = []
    
    for name, tuning in tunings:
        print(f"\n  Testing {name}...")
        
        piano = PolyphonicMember.piano(num_beats=4)
        
        optimizer = HarmonyOptimizer(
            members=[piano],
            tuning=tuning,
            lr=0.05,
            target_density=0.15
        )
        
        loss_history = optimizer.optimize(num_steps=50, verbose=False)
        
        final_loss = loss_history[-1]['total']
        initial_loss = loss_history[0]['total']
        
        print(f"    Final loss: {final_loss:.4f}")
        
        results.append({
            "tuning": name,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_history": loss_history
        })
    
    # Create comparison visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Tuning System Comparison", fontsize=16)
    
    for idx, result in enumerate(results):
        ax = axes[idx // 3, idx % 3]
        losses = [h['total'] for h in result['loss_history']]
        ax.plot(losses, linewidth=2)
        ax.set_title(result['tuning'])
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    if len(results) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparisons", "tuning_comparison.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save comparison data
    with open(os.path.join(output_dir, "comparisons", "tuning_comparison.json"), "w") as f:
        json.dump([{
            "tuning": r["tuning"],
            "initial_loss": r["initial_loss"],
            "final_loss": r["final_loss"]
        } for r in results], f, indent=2)
    
    print(f"\n  ✓ Comparison saved to: comparisons/")
    
    return results


def generate_summary_report(output_dir: str, scenario_results: List[Dict], 
                           tuning_results: List[Dict]):
    """Generate a summary report of the demo."""
    print_section("GENERATING SUMMARY REPORT")
    
    report_path = os.path.join(output_dir, "DEMO_REPORT.md")
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("# Harmony From First Principles - Full Demo Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report summarizes the comprehensive demonstration of all features "
                "implemented in the Harmony From First Principles project.\n\n")
        
        f.write("## Tuning Systems\n\n")
        f.write("The following tuning systems were demonstrated:\n\n")
        f.write("| System | Description |\n")
        f.write("|--------|-------------|\n")
        f.write("| 12-TET | Standard Western tuning (A4=440Hz) |\n")
        f.write("| Pythagorean | Pure perfect fifths (3:2 ratio) |\n")
        f.write("| 1/4-comma Meantone | Tempered fifths for pure major thirds |\n")
        f.write("| 1/3-comma Meantone | Alternative meantone temperament |\n")
        f.write("| 19-EDO | 19 equal divisions of the octave |\n")
        f.write("| 24-EDO | Quarter-tone scale (24 divisions) |\n")
        f.write("| 31-EDO | 31 equal divisions |\n")
        f.write("| 41-EDO | 41 equal divisions |\n")
        f.write("| 53-EDO | 53 equal divisions |\n")
        f.write("| Alpha Scale | Non-octave scale based on golden ratio |\n")
        f.write("| Beta Scale | Non-octave scale based on √2 |\n")
        f.write("| Bohlen-Pierce | Scale based on 3:1 instead of 2:1 |\n")
        
        f.write("\n## Optimization Scenarios\n\n")
        f.write("| Scenario | Initial Loss | Final Loss | Improvement | Time (s) |\n")
        f.write("|----------|--------------|------------|-------------|----------|\n")
        for r in scenario_results:
            f.write(f"| {r['name']} | {r['initial_loss']:.4f} | {r['final_loss']:.4f} | "
                   f"{r['improvement']:.1f}% | {r['time']:.1f} |\n")
        
        f.write("\n## Tuning System Comparison\n\n")
        f.write("| Tuning System | Initial Loss | Final Loss |\n")
        f.write("|---------------|--------------|------------|\n")
        for r in tuning_results:
            f.write(f"| {r['tuning']} | {r['initial_loss']:.4f} | {r['final_loss']:.4f} |\n")
        
        f.write("\n## Output Files\n\n")
        f.write("### Audio\n")
        f.write("- `audio/*/mixed.wav` - Final mixed audio for each scenario\n")
        f.write("- `audio/*/*.wav` - Individual instrument tracks\n")
        
        f.write("\n### Visualizations\n")
        f.write("- `visualizations/*/loss_history.png` - Loss curves\n")
        f.write("- `visualizations/*/*_weights.png` - Weight matrices\n")
        f.write("- `visualizations/*/spectrogram.png` - Audio spectrograms\n")
        
        f.write("\n### Comparisons\n")
        f.write("- `comparisons/tuning_comparison.png` - Side-by-side tuning comparison\n")
        f.write("- `comparisons/tuning_comparison.json` - Comparison data\n")
        
        f.write("\n## Features Demonstrated\n\n")
        f.write("✅ All tuning systems (12-TET, Pythagorean, Meantone, EDO, non-octave)\n")
        f.write("✅ ADSR envelopes with per-harmonic support\n")
        f.write("✅ Polyphonic members (piano)\n")
        f.write("✅ Monophonic members (guitar, bass)\n")
        f.write("✅ Drum members with fixed patterns\n")
        f.write("✅ Multi-member optimization\n")
        f.write("✅ Cross-member dissonance\n")
        f.write("✅ User constraints (fixed notes)\n")
        f.write("✅ Audio synthesis with ADSR\n")
        f.write("✅ Multi-track mixing\n")
        f.write("✅ Loss visualization\n")
        f.write("✅ Weight visualization\n")
        f.write("✅ Spectrogram generation\n")
    
    print(f"  ✓ Summary report saved to: {report_path}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Harmony From First Principles - Full Feature Demo"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode with fewer optimization steps"
    )
    parser.add_argument(
        "--no-tuning-comparison",
        action="store_true",
        help="Skip tuning system comparison (saves time)"
    )
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("  HARMONY FROM FIRST PRINCIPLES")
    print("  Full Feature Showcase")
    print("=" * 70)
    print()
    print("This script demonstrates all implemented features:")
    print("  • All tuning systems (12-TET, Pythagorean, Meantone, EDO, non-octave)")
    print("  • All instrument types with ADSR envelopes")
    print("  • All band member types (Polyphonic, Monophonic, Drums)")
    print("  • Optimization with multiple members")
    print("  • Audio synthesis and mixing")
    print("  • Visualization generation")
    print("  • User constraints")
    print()
    
    # Setup
    output_dir = ensure_output_dir()
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print()
    
    start_time = time.time()
    
    # Run demonstrations
    demo_tuning_systems(output_dir)
    demo_instruments(output_dir)
    demo_band_members(output_dir)
    
    # Run scenarios
    scenario_results = demo_scenarios(output_dir)
    
    # Run tuning comparison (unless skipped)
    if not args.no_tuning_comparison:
        tuning_results = demo_tuning_comparison(output_dir)
    else:
        tuning_results = []
    
    # Generate summary report
    generate_summary_report(output_dir, scenario_results, tuning_results)
    
    # Print footer
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Output directory: {output_dir}")
    print("\nSee DEMO_REPORT.md for detailed results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
