from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Tuple, List, Dict, Callable
import torch

from instruments import Instrument, get_instrument
from tuning_systems import TuningSystem, get_tuning_system

class Member(ABC):
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        tuning_system: TuningSystem,
        instrument_range: List[int],
        velocity: float,
        tick_duration: float,
        total_ticks: int,
        ticks_per_note: int,
        hp: Dict[str, Any] = {},
        initial_weights: Optional[torch.Tensor] = None
    ):
        self.name = name
        self.instrument = instrument
        self.tuning_system = tuning_system
        self.instrument_range = instrument_range
        self.velocity = velocity
        # Check if instrument_range is two ints, then calculate num_keys
        if len(instrument_range) != 2:
            raise ValueError(f"instrument_range must be a list of two ints. Got {instrument_range}.")
        num_keys = instrument_range[1] - instrument_range[0] + 1
        self.num_keys = num_keys

        # Check if total_ticks is a multiple of ticks_per_note, then calculate num_notes
        if total_ticks % ticks_per_note != 0:
            raise ValueError(f"total_ticks must be a multiple of ticks_per_note. Got {total_ticks} total ticks and {ticks_per_note} ticks per note.")
        self.tick_duration = tick_duration
        self.total_ticks = total_ticks
        self.ticks_per_note = ticks_per_note
        num_notes = total_ticks // ticks_per_note
        self.num_notes = num_notes
        
        self.hp = hp
        # Initialize or use provided weights
        if initial_weights is not None:
            if initial_weights.shape != (num_keys, num_notes):
                raise ValueError(f"initial_weights must have shape ({num_keys}, {num_notes}). Got {initial_weights.shape}."
                )
            self.weights = torch.nn.Parameter(initial_weights)
        else:
            self.initialize()
        
        # Keep painted constraints out of nn.Parameter
        self.painted_weights = torch.zeros((num_keys, num_notes))
        self.painted_mask = torch.zeros((num_keys, num_notes), dtype=torch.bool)
    
    # Helpers for commonly used values
    def note_duration(self) -> float:
        return self.tick_duration * self.ticks_per_note
    
    def total_notes(self) -> int:
        return self.total_ticks // self.ticks_per_note
    
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def forward(self, x: Any) -> torch.Tensor:
        # x exists so this looks like a fake nn.Module but x is not used
        pass

    def get_effective_weights(self) -> torch.Tensor:
        """
        Get effective weights combining optimizable weights and painted constraints.
        
        Returns a tensor where painted positions have the user-specified values,
        and unpainted positions have the learned weights. Painted values are
        detached so gradients only flow to the optimizable weights.
        
        Returns:
            Tensor of shape (num_keys, num_notes)
        """
        # Painted weights are buffers (not Parameters), so they're naturally detached
        # Use torch.where for a differentiable selection
        # For painted positions: use painted_weights (detached)
        # For unpainted positions: use self.weights (gets gradients)
        effective = torch.where(
            self.painted_mask,
            self.painted_weights, # This is a buffer, no gradients
            self.weights          # This is a Parameter, gradients flow here
        )
        return effective

    def paint_weights(self, newly_painted: torch.Tensor):
        """
        Paint fixed constraint values at specific positions.
        
        Args:
            newly_painted: Tensor of shape (num_keys, num_notes) with values to fix.
                          Non-zero values will be painted (fixed during optimization).
        """
        with torch.no_grad():
            new_mask = (newly_painted != 0.0)
            self.painted_weights[new_mask] = newly_painted[new_mask]
            self.painted_mask[new_mask] = True

    def clear_paint(self):
        """Clear all painted constraints."""
        with torch.no_grad():
            self.painted_weights.zero_()
            self.painted_mask.zero_()

class PolyphonicMember(Member):
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        tuning_system: TuningSystem,
        instrument_range: List[int],
        velocity: float,
        tick_duration: float,
        total_ticks: int,
        ticks_per_note: int,
        hp: Dict[str, Any] = {},
        initial_weights: Optional[torch.Tensor] = None
    ):
        super().__init__(
            name,
            instrument,
            tuning_system,
            instrument_range,
            velocity,
            tick_duration,
            total_ticks,
            ticks_per_note,
            hp,
            initial_weights
        )
        # The super init will call PolyphonicMember.initialize()

    def initialize(self):
        # PolyphonicMember uses ReLU
        # Once dead, a weight stays dead
        # Initialize weights in (0.0, 0.1]
        self.weights = torch.nn.Parameter(
            1.0 - torch.rand((self.num_keys, self.num_notes))
        )
    
    def forward(self, x: Any) -> torch.Tensor:
        # Apply ReLU to effective weights (combines optimizable + painted constraints)
        return torch.relu(self.get_effective_weights())

class MonophonicMember(Member):
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        tuning_system: TuningSystem,
        instrument_range: List[int],
        velocity: float,
        tick_duration: float,
        total_ticks: int,
        ticks_per_note: int,
        hp: Dict[str, Any] = {},
        initial_weights: Optional[torch.Tensor] = None
    ):
        super().__init__(
            name,
            instrument,
            tuning_system,
            instrument_range,
            velocity,
            tick_duration,
            total_ticks,
            ticks_per_note,
            hp,
            initial_weights
        )
        # The super init will call MonophonicMember.initialize()

    def initialize(self):
        # MonophonicMember uses Gumbel-softmax
        # Initialize weights in N(0, 1)
        self.weights = torch.nn.Parameter(
            torch.randn((self.num_keys, self.num_notes))
        )
    
    def forward(self, x: Any) -> torch.Tensor:
        # Gumbel-softmax (straight-through estimator)
        # Forward: argmax, Backward: softmax with temperature
        # Use effective weights (combines optimizable + painted constraints)
        logits = self.get_effective_weights()  # (keys, notes)
        
        # During forward pass for loss computation, use soft probabilities
        # Temperature can be adjusted - lower = more discrete
        temperature = self.hp.get('gumbel_temperature', 0.5)
        
        # Apply softmax across keys dimension
        probs = torch.softmax(logits / temperature, dim=0)
        
        # Straight-through estimator: forward uses hard, backward uses soft
        hard = torch.zeros_like(logits)
        max_indices = torch.argmax(logits, dim=0)
        hard.scatter_(0, max_indices.unsqueeze(0), 1.0)
        
        # Use hard for forward, soft for backward
        activation = hard + (probs - probs.detach())
        return activation

# Use a registry pattern to turn str into default objects
_MEMBER_REGISTRY = {}

def register_member(
    name: str,
    factory: Callable[..., Member]
):
    _MEMBER_REGISTRY[name] = factory

def get_member(
    name_or_instance: Union[str, Member],
    **kwargs
) -> Member:
    if isinstance(name_or_instance, Member):
        return name_or_instance
    if name_or_instance not in _MEMBER_REGISTRY:
        raise ValueError(f"Unknown MEMBER: {name_or_instance}")
    return _MEMBER_REGISTRY[name_or_instance](**kwargs)

# Register defaults

# Helper function to cut down on line bloat (it's all the same 4 parameters man)
def register_member_helper(name: str, instrument_name: str, instrument_range: List[int], polyphonic: bool = True):
    member_class = PolyphonicMember if polyphonic else MonophonicMember
    register_member(name, lambda velocity, tick_duration, total_ticks, ticks_per_note, **kwargs: member_class(
        name=name,
        instrument=get_instrument(instrument_name),
        tuning_system=get_tuning_system("12-TET"),
        instrument_range=instrument_range,
        velocity=velocity,
        tick_duration=tick_duration,
        total_ticks=total_ticks,
        ticks_per_note=ticks_per_note,
        **kwargs
    ))

# 🎸🎹🥁🍜 4=1

# A 6 string guitar (E2 to E4) with 24 frets (up to E6)
register_member_helper("guitar", "guitar", [40, 88], False)

# A standard 88 key piano (A0 to C8)
# Despite the piano being in this section, the piano is more associated with 🎼 Yoisaki Kanade or ☕️ Aoyagi Toya
# 🎹 Tenma Saki's standard instrument is specifically a yellow synth
register_member_helper("piano", "piano", [21, 108], True)

# Drum kit - uses MIDI note numbers
# This is a special member that uses PercussionInstrument
# 35 (Acoustic Bass Drum), 38 (Acoustic Snare), 42 (Closed Hi Hat)
register_member_helper("drums", "percussion", [35, 42], True)

# A 4 string bass (E1 to G3) limited to 12 frets (up to G4). This should keep the optimizer from trying to use the bass too high
register_member_helper("bass", "bass", [28, 67], False)

# Keyboards

# Rhodes/Electric Piano - jazz/funk/soul keyboard
# E1 to E6 (full 5-octave Rhodes range)
register_member_helper("rhodes", "rhodes", [28, 88], True)

# Organ - classical/rock/gospel keyboard with multiple ranks
# C2 to C7 (standard organ keyboard)
register_member_helper("organ", "organ", [36, 96], False)

# Bowed Strings

# Violin - can be polyphonic (string section) or monophonic (solo)
# G3 to E7 (standard violin range)
register_member_helper("violin_section", "violin", [55, 103], True)
register_member_helper("violin", "violin", [55, 103], False)

# Cello - warm lower range
# C2 to E5 (standard cello range)
register_member_helper("cello_section", "cello", [36, 76], True)
register_member_helper("cello", "cello", [36, 76], False)

# Brass

# Trumpet - bright brass instrument
# F#3 to D6 (standard trumpet range)
register_member_helper("trumpet_section", "trumpet", [54, 86], True)
register_member_helper("trumpet", "trumpet", [54, 86], False)

# Woodwinds

# Flute - breathy woodwind
# C4 to C7 (standard flute range)
register_member_helper("flute_section", "flute", [60, 96], True)
register_member_helper("flute", "flute", [60, 96], False)

# Clarinet - hollow-sounding woodwind
# D3 to G6 (standard clarinet range)
register_member_helper("clarinet_section", "clarinet", [50, 91], True)
register_member_helper("clarinet", "clarinet", [50, 91], False)

# Saxophone - bright reed instrument
# Bb3 to D6 (alto sax range, transposed to concert pitch)
register_member_helper("saxophone_section", "saxophone", [49, 87], True)
register_member_helper("saxophone", "saxophone", [49, 87], False)

# Synth / Electronic

# A 128 key pure sine tone instrument that supports all MIDI notes
register_member_helper("midi_pad", "sine", [0, 127], True)
register_member_helper("midi_lead", "sine", [0, 127], False)

# The above should be considered test instruments.
# For actual usage, keep instrument ranges in [21, 108] (A0 to C8) because the optimizer will boost the really high frequencies (above Nyquist) since they can't cause dissonance

# Saw Lead - classic subtractive synthesis - harmonics 1, 2, 3, 4, ...
register_member_helper("saw_pad", "saw", [21, 108], True)
register_member_helper("saw_lead", "saw", [21, 108], False)

# Square Lead - chiptune/8-bit style - harmonics 1, 3, 5, 7, ...
register_member_helper("square_pad", "square", [21, 108], True)
register_member_helper("square_lead", "square", [21, 108], False)

# Idiophones

# Music Box - clear bell-like tones
# C4 to C7 (typical music box range)
register_member_helper("music_box", "music_box", [60, 96], True)
register_member_helper("music_box_lead", "music_box", [60, 96], False)

# Vibraphone - struck metal bars with tremolo
# F3 to F6 (standard vibraphone range)
register_member_helper("vibraphone", "vibraphone", [54, 89], True)
register_member_helper("vibraphone_lead", "vibraphone", [54, 89], False)

# Choir / Voice

# C3 to C5. The lowest bass can go to E2 and the highest soprano can go to C6, but 1) the optimizer doesn't handle instruments with massive ranges well and 2) don't expect a trained choir

# Choir "ooh" - /u/ rounded vowel with low first formant
register_member_helper("choir_ooh", "voice_ooh", [48, 72], True)
register_member_helper("voice_ooh", "voice_ooh", [48, 72], False)

# Choir "aah" - /a/ open vowel with higher first formant
register_member_helper("choir_aah", "voice_aah", [48, 72], True)
register_member_helper("voice_aah", "voice_aah", [48, 72], False)

# Choir "eeh" - /i/ bright vowel with very high F2
register_member_helper("choir_eeh", "voice_eeh", [48, 72], True)
register_member_helper("voice_eeh", "voice_eeh", [48, 72], False)
