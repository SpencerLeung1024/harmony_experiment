"""
Gradio web interface for Harmony From First Principles.

This module provides a web-based UI for the harmony optimizer, allowing users to:
- Select tuning systems and band members
- Configure optimization parameters
- Add user constraints (fixed notes)
- Run optimization and view results
- Listen to generated audio
- Download audio and visualizations
"""

from typing import List, Dict, Optional, Tuple, Any, Callable
import os
import json
import numpy as np
import torch
import gradio as gr

from .tuning import (
    TuningSystem, TwelveTET, PythagoreanTuning, 
    MeantoneTuning, EDOSystem, NonOctaveSystem
)
from .band import BandMember, PolyphonicMember, MonophonicMember, DrumMember
from .constraints import ConstraintSet, UserConstraint
from .optimizer import HarmonyOptimizer
from .visualization import (
    plot_weights, plot_loss_history, plot_dissonance_matrix,
    create_weight_piano_roll, save_audio, save_weights_plot,
    fig_to_numpy, PITCH_CLASSES
)


# Tuning system options
TUNING_OPTIONS = {
    "12-TET": TwelveTET,
    "Pythagorean": PythagoreanTuning,
    "Quarter-comma Meantone": lambda: MeantoneTuning(comma_fraction=0.25),
    "Third-comma Meantone": lambda: MeantoneTuning(comma_fraction=1/3),
    "19-EDO": lambda: EDOSystem(divisions=19),
    "24-EDO": lambda: EDOSystem(divisions=24),
    "31-EDO": lambda: EDOSystem(divisions=31),
    "41-EDO": lambda: EDOSystem(divisions=41),
    "53-EDO": lambda: EDOSystem(divisions=53),
    "Alpha Scale (Golden Ratio)": NonOctaveSystem.alpha_scale,
    "Beta Scale (Sqrt 2)": NonOctaveSystem.beta_scale,
    "Bohlen-Pierce (3:1)": NonOctaveSystem.bohlen_pierce,
}


class GradioInterface:
    """Gradio web interface for the harmony optimizer.
    
    Provides an intuitive UI for non-technical users to:
    - Configure tuning systems and band members
    - Set optimization parameters
    - Add note constraints
    - Run optimization
    - View and download results
    
    Attributes:
        optimizer: Current HarmonyOptimizer instance
        members: Current list of band members
        constraints: Current ConstraintSet
        tuning: Current tuning system
        audio: Most recent generated audio
        loss_history: History of loss values from optimization
    """
    
    def __init__(self):
        """Initialize the Gradio interface."""
        self.optimizer: Optional[HarmonyOptimizer] = None
        self.members: List[BandMember] = []
        self.constraints = ConstraintSet()
        self.tuning: TuningSystem = TwelveTET()
        self.audio: Optional[np.ndarray] = None
        self.loss_history: List[Dict[str, float]] = []
        
        # Default parameters
        self.default_steps = 100
        self.default_lr = 0.02
        self.default_density = 0.15
        self.default_temporal_decay = 0.3
        
    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio Blocks interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Harmony From First Principles") as demo:
            gr.Markdown(
                """# 🎵 Harmony From First Principles
                
                Generate harmonious music using physics-based optimization.
                
                This tool optimizes note selections to minimize dissonance while 
                maintaining musical interest across multiple instruments and tuning systems.
                """
            )
            
            # State storage for constraints
            constraint_state = gr.State(value=[])
            
            with gr.Tab("🎛️ Configuration"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Tuning System")
                        tuning_dropdown = gr.Dropdown(
                            choices=list(TUNING_OPTIONS.keys()),
                            value="12-TET",
                            label="Tuning System",
                            info="Select the musical tuning system for all instruments"
                        )
                        
                        gr.Markdown("### Band Members")
                        piano_checkbox = gr.Checkbox(
                            label="🎹 Piano (Polyphonic)",
                            value=True,
                            info="Can play multiple notes simultaneously (chords)"
                        )
                        guitar_checkbox = gr.Checkbox(
                            label="🎸 Guitar (Monophonic)",
                            value=False,
                            info="Plays one note at a time"
                        )
                        bass_checkbox = gr.Checkbox(
                            label="🎸 Bass (Monophonic)",
                            value=False,
                            info="Low-pitched monophonic instrument"
                        )
                        drums_checkbox = gr.Checkbox(
                            label="🥁 Drums",
                            value=False,
                            info="Fixed kick/snare pattern on beats 1,3 and 2,4"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Optimization Parameters")
                        steps_slider = gr.Slider(
                            minimum=10, maximum=500, step=10, value=100,
                            label="Optimization Steps",
                            info="More steps = better convergence but slower"
                        )
                        lr_slider = gr.Slider(
                            minimum=0.001, maximum=0.1, step=0.001, value=0.02,
                            label="Learning Rate",
                            info="Step size for gradient descent"
                        )
                        density_slider = gr.Slider(
                            minimum=0.05, maximum=0.5, step=0.05, value=0.15,
                            label="Target Note Density",
                            info="Target proportion of active notes (0-1)"
                        )
                        
                        gr.Markdown("### Audio Settings")
                        duration_slider = gr.Slider(
                            minimum=2, maximum=10, step=1, value=4,
                            label="Duration (seconds)",
                            info="Length of generated audio"
                        )
                        sample_rate_dropdown = gr.Dropdown(
                            choices=[22050, 44100],
                            value=22050,
                            label="Sample Rate",
                            info="Higher = better quality but larger files"
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Loss Weights")
                        with gr.Row():
                            within_weight = gr.Slider(
                                minimum=0, maximum=5, step=0.1, value=1.0,
                                label="Within-member Dissonance",
                                info="Penalty for dissonant notes played simultaneously"
                            )
                            cross_weight = gr.Slider(
                                minimum=0, maximum=5, step=0.1, value=1.0,
                                label="Cross-member Dissonance",
                                info="Penalty for dissonance between different instruments"
                            )
                        with gr.Row():
                            temporal_weight = gr.Slider(
                                minimum=0, maximum=2, step=0.1, value=0.5,
                                label="Temporal Dissonance",
                                info="Penalty for dissonance between adjacent beats"
                            )
                            density_weight = gr.Slider(
                                minimum=0, maximum=20, step=1, value=10.0,
                                label="Density Penalty",
                                info="Penalty for deviating from target density"
                            )
            
            with gr.Tab("🎹 Constraints"):
                gr.Markdown(
                    """### User Constraints (Fixed Notes)
                    
                    Add fixed notes that will influence the optimization but won't be changed.
                    This is like "partial denoising" - you're guiding the AI toward certain harmonies.
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        constraint_member = gr.Dropdown(
                            choices=["piano", "guitar", "bass"],
                            value="piano",
                            label="Member",
                            info="Which instrument to constrain"
                        )
                        constraint_beat = gr.Number(
                            value=0, minimum=0, maximum=15, step=1,
                            label="Beat Index",
                            info="Which beat (0-indexed)"
                        )
                        constraint_note = gr.Dropdown(
                            choices=self._get_note_choices(),
                            value="C4",
                            label="Note",
                            info="MIDI note name"
                        )
                        add_constraint_btn = gr.Button("➕ Add Constraint", variant="primary")
                    
                    with gr.Column(scale=2):
                        constraints_display = gr.Dataframe(
                            headers=["Member", "Beat", "Note", "MIDI Key"],
                            label="Current Constraints",
                            interactive=False
                        )
                        clear_constraints_btn = gr.Button("🗑️ Clear All Constraints")
            
            with gr.Tab("🎵 Generate"):
                with gr.Row():
                    with gr.Column():
                        optimize_btn = gr.Button("🚀 Optimize", variant="primary", size="lg")
                        preview_btn = gr.Button("⚡ Quick Preview (50 steps)", variant="secondary")
                        
                        gr.Markdown("### Progress")
                        progress_bar = gr.Progress()
                        current_loss = gr.Textbox(
                            label="Current Loss",
                            value="Not started",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Results")
                        audio_output = gr.Audio(
                            label="Generated Audio",
                            type="numpy",
                            autoplay=False
                        )
                        
                        with gr.Row():
                            download_audio_btn = gr.Button("💾 Download Audio")
                            download_viz_btn = gr.Button("📊 Download Visualizations")
            
            with gr.Tab("📊 Visualizations"):
                with gr.Row():
                    piano_roll_plot = gr.Image(
                        label="Piano Roll",
                        type="numpy"
                    )
                    weights_plot = gr.Image(
                        label="Weights Heatmap",
                        type="numpy"
                    )
                
                loss_plot = gr.Image(
                    label="Loss History",
                    type="numpy"
                )
            
            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    """
                    ## Harmony From First Principles
                    
                    This project generates harmonious music through physics-based optimization.
                    
                    ### How it works
                    
                    1. **Dissonance Calculation**: Based on sensory dissonance theory, the algorithm 
                       calculates how dissonant any two frequencies are based on their harmonic content.
                    
                    2. **Optimization**: Using PyTorch's gradient descent, the algorithm optimizes 
                       which notes each instrument plays to minimize total dissonance while 
                       maintaining musical interest.
                    
                    3. **Multiple Band Members**: Piano (polyphonic), Guitar (monophonic), 
                       Bass (monophonic), and Drums work together.
                    
                    4. **Alternative Tunings**: Explore different musical systems including 
                       12-TET, Pythagorean, Meantone, and various EDO (equal division of octave) tunings.
                    
                    ### Tips
                    
                    - Start with Piano only to understand the basics
                    - Add Guitar and Bass for more complex harmonies
                    - Use constraints to guide the AI toward specific chords
                    - Experiment with different tuning systems for unique sounds
                    - Higher optimization steps give better results but take longer
                    
                    ### Credits
                    
                    Based on "Harmony From First Principles" - exploring the physics of musical harmony.
                    """
                )
            
            # Event handlers
            tuning_dropdown.change(
                fn=self._on_tuning_change,
                inputs=[tuning_dropdown],
                outputs=[]
            )
            
            add_constraint_btn.click(
                fn=self._on_add_constraint,
                inputs=[constraint_state, constraint_member, constraint_beat, constraint_note],
                outputs=[constraint_state, constraints_display]
            )
            
            clear_constraints_btn.click(
                fn=self._on_clear_constraints,
                inputs=[],
                outputs=[constraint_state, constraints_display]
            )
            
            optimize_btn.click(
                fn=self._on_optimize,
                inputs=[
                    tuning_dropdown, piano_checkbox, guitar_checkbox, bass_checkbox, drums_checkbox,
                    steps_slider, lr_slider, density_slider, duration_slider, sample_rate_dropdown,
                    within_weight, cross_weight, temporal_weight, density_weight,
                    constraint_state
                ],
                outputs=[
                    audio_output, piano_roll_plot, weights_plot, loss_plot, current_loss
                ]
            )
            
            preview_btn.click(
                fn=self._on_preview,
                inputs=[
                    tuning_dropdown, piano_checkbox, guitar_checkbox, bass_checkbox, drums_checkbox,
                    duration_slider, sample_rate_dropdown,
                    within_weight, cross_weight, temporal_weight, density_weight,
                    constraint_state
                ],
                outputs=[
                    audio_output, piano_roll_plot, weights_plot, loss_plot, current_loss
                ]
            )
            
            download_audio_btn.click(
                fn=self._on_download_audio,
                inputs=[],
                outputs=[gr.File(label="Download Audio")]
            )
            
            download_viz_btn.click(
                fn=self._on_download_viz,
                inputs=[],
                outputs=[gr.File(label="Download Visualizations")]
            )
            
        return demo
    
    def _get_css(self) -> str:
        """Get custom CSS for the interface."""
        return """
        .gradio-container {
            font-family: 'Inter', sans-serif;
        }
        """
    
    def _get_note_choices(self) -> List[str]:
        """Get list of note choices for constraint dropdown."""
        notes = []
        for octave in range(0, 10):
            for pc in PITCH_CLASSES:
                notes.append(f"{pc}{octave}")
        return notes
    
    def _note_to_midi(self, note_name: str) -> int:
        """Convert note name (e.g., 'C4') to MIDI key number."""
        pc_names = PITCH_CLASSES
        pc = note_name[:-1]  # Remove octave
        octave = int(note_name[-1])
        pc_idx = pc_names.index(pc)
        return (octave + 1) * 12 + pc_idx
    
    def _on_tuning_change(self, tuning_name: str) -> None:
        """Handle tuning system change."""
        tuning_class = TUNING_OPTIONS[tuning_name]
        self.tuning = tuning_class()
    
    def _on_add_constraint(
        self,
        constraint_list: List[Dict],
        member: str,
        beat: int,
        note: str
    ) -> Tuple[List[Dict], Dict]:
        """Handle adding a constraint."""
        midi_key = self._note_to_midi(note)
        
        # Determine key offset based on member
        key_offsets = {"piano": 21, "guitar": 40, "bass": 28}
        key_offset = key_offsets.get(member, 0)
        key_index = midi_key - key_offset
        
        constraint_list = constraint_list or []
        constraint_list.append({
            "member": member,
            "beat": int(beat),
            "note": note,
            "midi_key": midi_key,
            "key_index": key_index
        })
        
        # Create dataframe
        df_data = {
            "Member": [c["member"] for c in constraint_list],
            "Beat": [c["beat"] for c in constraint_list],
            "Note": [c["note"] for c in constraint_list],
            "MIDI Key": [c["midi_key"] for c in constraint_list]
        }
        
        return constraint_list, df_data
    
    def _on_clear_constraints(self) -> Tuple[List, Dict]:
        """Handle clearing all constraints."""
        return [], {"Member": [], "Beat": [], "Note": [], "MIDI Key": []}
    
    def _create_members(
        self,
        tuning_name: str,
        piano: bool,
        guitar: bool,
        bass: bool,
        drums: bool,
        num_beats: int = 4
    ) -> List[BandMember]:
        """Create band members based on selections."""
        tuning_class = TUNING_OPTIONS[tuning_name]
        tuning = tuning_class()
        
        members = []
        if piano:
            members.append(PolyphonicMember.piano(num_beats=num_beats, tuning=tuning))
        if guitar:
            members.append(MonophonicMember.guitar(num_beats=num_beats, tuning=tuning))
        if bass:
            members.append(MonophonicMember.bass(num_beats=num_beats, tuning=tuning))
        if drums:
            members.append(DrumMember.standard_rock(num_beats=num_beats, enabled=True))
        
        return members
    
    def _create_constraints(self, constraint_list: List[Dict]) -> ConstraintSet:
        """Create ConstraintSet from constraint list."""
        constraint_set = ConstraintSet()
        
        for c in constraint_list:
            constraint = UserConstraint(
                member_name=c["member"],
                beat_index=c["beat"],
                key_indices=c["key_index"],
                strengths=1.0
            )
            constraint_set.add_constraint(constraint)
        
        return constraint_set
    
    def _on_optimize(
        self,
        tuning_name: str,
        piano: bool,
        guitar: bool,
        bass: bool,
        drums: bool,
        steps: int,
        lr: float,
        density: float,
        duration: float,
        sample_rate: int,
        within_weight: float,
        cross_weight: float,
        temporal_weight: float,
        density_weight: float,
        constraint_list: List[Dict]
    ) -> Tuple:
        """Run full optimization."""
        return self._run_optimization(
            tuning_name=tuning_name,
            piano=piano,
            guitar=guitar,
            bass=bass,
            drums=drums,
            steps=steps,
            lr=lr,
            density=density,
            duration=duration,
            sample_rate=sample_rate,
            within_weight=within_weight,
            cross_weight=cross_weight,
            temporal_weight=temporal_weight,
            density_weight=density_weight,
            constraint_list=constraint_list
        )
    
    def _on_preview(
        self,
        tuning_name: str,
        piano: bool,
        guitar: bool,
        bass: bool,
        drums: bool,
        duration: float,
        sample_rate: int,
        within_weight: float,
        cross_weight: float,
        temporal_weight: float,
        density_weight: float,
        constraint_list: List[Dict]
    ) -> Tuple:
        """Run quick preview optimization (fewer steps)."""
        return self._run_optimization(
            tuning_name=tuning_name,
            piano=piano,
            guitar=guitar,
            bass=bass,
            drums=drums,
            steps=50,
            lr=0.03,
            density=0.15,
            duration=duration,
            sample_rate=sample_rate,
            within_weight=within_weight,
            cross_weight=cross_weight,
            temporal_weight=temporal_weight,
            density_weight=density_weight,
            constraint_list=constraint_list
        )
    
    def _run_optimization(
        self,
        tuning_name: str,
        piano: bool,
        guitar: bool,
        bass: bool,
        drums: bool,
        steps: int,
        lr: float,
        density: float,
        duration: float,
        sample_rate: int,
        within_weight: float,
        cross_weight: float,
        temporal_weight: float,
        density_weight: float,
        constraint_list: List[Dict]
    ) -> Tuple:
        """Run the optimization and return results."""
        # Check that at least one member is selected
        if not any([piano, guitar, bass, drums]):
            gr.Warning("Please select at least one band member!")
            return None, None, None, None, "Error: No members selected"
        
        # Create members
        num_beats = 4
        self.members = self._create_members(
            tuning_name, piano, guitar, bass, drums, num_beats
        )
        
        if len(self.members) == 0:
            return None, None, None, None, "Error: No members created"
        
        # Create constraints
        self.constraints = self._create_constraints(constraint_list)
        
        # Loss weights
        loss_weights = {
            'within': within_weight,
            'cross': cross_weight,
            'temporal': temporal_weight,
            'density': density_weight,
            'sparsity': 1.0,
            'range': 1.0,
            'interval_jump': 0.5
        }
        
        # Create optimizer
        self.optimizer = HarmonyOptimizer(
            members=self.members,
            tuning=None,  # Members already have tuning
            constraints=self.constraints,
            loss_weights=loss_weights,
            lr=lr,
            target_density=density,
            temporal_decay=0.3
        )
        
        # Precompute dissonance
        self.optimizer.precompute_dissonance()
        
        # Run optimization with progress tracking
        self.loss_history = []
        
        def progress_callback(step: int, loss_dict: Dict[str, float]):
            self.loss_history.append(loss_dict)
            # Progress is handled by Gradio's progress bar
        
        result = self.optimizer.optimize(
            steps=steps,
            callback=progress_callback,
            verbose=False
        )
        
        # Generate audio
        self.audio = self.optimizer.get_audio(
            duration=duration,
            sample_rate=sample_rate
        )
        
        # Create visualizations
        try:
            piano_roll_fig = create_weight_piano_roll(
                self.members, self.constraints, title="Piano Roll"
            )
            piano_roll_img = fig_to_numpy(piano_roll_fig)
        except Exception as e:
            print(f"Error creating piano roll: {e}")
            piano_roll_img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        try:
            # Get first pitched member for weights plot
            pitched_members = [m for m in self.members if not isinstance(m, DrumMember)]
            if pitched_members:
                weights_fig = plot_weights(
                    pitched_members[0].weights,
                    title=f"{pitched_members[0].name} Weights",
                    member=pitched_members[0]
                )
                weights_img = fig_to_numpy(weights_fig)
            else:
                weights_img = np.zeros((400, 600, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error creating weights plot: {e}")
            weights_img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        try:
            loss_fig = plot_loss_history(self.loss_history, title="Loss History")
            loss_img = fig_to_numpy(loss_fig)
        except Exception as e:
            print(f"Error creating loss plot: {e}")
            loss_img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        final_loss = result['final_loss']
        
        return (
            (sample_rate, self.audio),
            piano_roll_img,
            weights_img,
            loss_img,
            f"Final Loss: {final_loss:.4f}"
        )
    
    def _on_download_audio(self) -> Optional[str]:
        """Handle audio download."""
        if self.audio is None:
            gr.Warning("No audio generated yet! Please run optimization first.")
            return None
        
        # Save to temporary file
        temp_path = "output/generated_audio.wav"
        os.makedirs("output", exist_ok=True)
        save_audio(self.audio, temp_path, sample_rate=22050)
        
        return temp_path
    
    def _on_download_viz(self) -> Optional[str]:
        """Handle visualization download."""
        if not self.members:
            gr.Warning("No members to visualize! Please run optimization first.")
            return None
        
        # Save to file
        temp_path = "output/visualization.png"
        os.makedirs("output", exist_ok=True)
        save_weights_plot(self.members, temp_path, self.constraints)
        
        return temp_path


# ==================== VERIFICATION TESTS ====================

if __name__ == "__main__":
    print("=" * 60)
    print("GRADIO INTERFACE VERIFICATION")
    print("=" * 60)
    
    # Test 1: Create interface
    print("\n1. Testing interface creation:")
    interface = GradioInterface()
    demo = interface.create_interface()
    print(f"   Interface type: {type(demo)}")
    print("   ✓ Interface created successfully")
    
    # Test 2: Test note conversion
    print("\n2. Testing note conversion:")
    test_notes = ["C4", "A4", "F#5"]
    for note in test_notes:
        midi = interface._note_to_midi(note)
        print(f"   {note} -> MIDI {midi}")
    print("   ✓ Note conversion working")
    
    # Test 3: Test member creation
    print("\n3. Testing member creation:")
    members = interface._create_members(
        "12-TET", piano=True, guitar=True, bass=False, drums=False
    )
    print(f"   Created {len(members)} members:")
    for m in members:
        print(f"     - {m.name}: {m.num_keys} keys, {m.num_beats} beats")
    print("   ✓ Members created successfully")
    
    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("Run with: interface.launch()")
    print("=" * 60)
