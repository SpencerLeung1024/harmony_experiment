"""
Harmony From First Principles - Musical optimization framework.

This package provides tools for exploring harmony through optimization,
supporting arbitrary tuning systems and instrument timbres.

Modules:
    tuning: Various tuning systems (12-TET, Pythagorean, Meantone, EDO, etc.)
    instruments: Instrument models with ADSR envelopes and harmonic profiles
    dissonance: Dissonance calculation for arbitrary tunings and timbres
    band: Band member system (polyphonic, monophonic, drums)
    losses: Multi-member loss functions
    constraints: User constraints and fixed patterns
    optimizer: Main harmony optimization orchestration
    synthesis: Audio synthesis with ADSR envelopes
    mixer: Audio mixing with gain control and limiting
    visualization: Plotting and visualization utilities
    ui: Gradio web interface
"""

from .tuning import (
    TuningSystem,
    TwelveTET,
    PythagoreanTuning,
    MeantoneTuning,
    EDOSystem,
    NonOctaveSystem,
)

from .instruments import (
    ADSR,
    Instrument,
)

from .dissonance import (
    DissonanceCalculator,
)

from .band import (
    BandMember,
    PolyphonicMember,
    MonophonicMember,
    DrumMember,
)

from .losses import (
    LossFunction,
)

from .constraints import (
    UserConstraint,
    ConstraintSet,
    DrumPatternConstraint,
)

from .optimizer import (
    HarmonyOptimizer,
)

from .synthesis import (
    AudioSynthesizer,
)

from .mixer import (
    AudioMixer,
)

from .visualization import (
    plot_weights,
    plot_spectrogram,
    plot_dissonance_matrix,
    plot_loss_history,
    create_weight_piano_roll,
    color_weights_by_pitch_class,
    save_audio,
    save_weights_plot,
)

__all__ = [
    # Tuning systems
    "TuningSystem",
    "TwelveTET",
    "PythagoreanTuning",
    "MeantoneTuning",
    "EDOSystem",
    "NonOctaveSystem",
    # Instruments
    "ADSR",
    "Instrument",
    # Dissonance
    "DissonanceCalculator",
    # Band members
    "BandMember",
    "PolyphonicMember",
    "MonophonicMember",
    "DrumMember",
    # Losses
    "LossFunction",
    # Constraints
    "UserConstraint",
    "ConstraintSet",
    "DrumPatternConstraint",
    # Optimizer
    "HarmonyOptimizer",
    # Synthesis
    "AudioSynthesizer",
    # Mixing
    "AudioMixer",
    # Visualization
    "plot_weights",
    "plot_spectrogram",
    "plot_dissonance_matrix",
    "plot_loss_history",
    "create_weight_piano_roll",
    "color_weights_by_pitch_class",
    "save_audio",
    "save_weights_plot",
]
