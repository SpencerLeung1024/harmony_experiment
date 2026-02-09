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
            self.painted_weights = torch.zeros_like(self.weights)  # Track painted values
            self.painted_mask = (self.painted_weights != 0.0)  # Mask for painted weights
        else:
            self.initialize()
        
        # Use register_post_accumulate_grad_hook to zero gradients of painted weights (they will not change)
        # RuntimeError: Tensor post accumulate grad hooks should return None.
        #self.handle = self.weights.register_post_accumulate_grad_hook(lambda p: p.grad.masked_fill_(self.painted_mask, 0.0))
    
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

    def paint_weights(self, newly_painted_weights: torch.Tensor):
        new_mask = (newly_painted_weights != 0.0)
        self.painted_weights[new_mask] = newly_painted_weights[new_mask]
        self.painted_mask = (self.painted_weights != 0.0)
        # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
        #self.weights[self.painted_mask] = self.painted_weights[self.painted_mask]

class PolyphonicMember(Member):
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        tuning_system: TuningSystem,
        instrument_range: List[int],
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
        self.painted_weights = torch.zeros_like(self.weights)
        self.painted_mask = (self.painted_weights != 0.0)
    
    def forward(self, x: Any) -> torch.Tensor:
        return torch.relu(self.weights)

class MonophonicMember(Member):
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        tuning_system: TuningSystem,
        instrument_range: List[int],
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
        self.painted_weights = torch.zeros_like(self.weights)
        self.painted_mask = (self.painted_weights != 0.0)
    
    def forward(self, x: Any) -> torch.Tensor:
        # Gumbel-softmax (straight-through estimator)
        # Forward: argmax, Backward: softmax with temperature
        logits = self.weights  # (keys, notes)
        
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

# A standard 88 key piano
register_member("piano", lambda tick_duration, total_ticks, ticks_per_note, **kwargs: PolyphonicMember(
    name="piano",
    instrument=get_instrument("piano"),
    tuning_system=get_tuning_system("12-TET"),
    instrument_range=[21, 108],
    tick_duration=tick_duration,
    total_ticks=total_ticks,
    ticks_per_note=ticks_per_note,
    **kwargs
))

# A 6 string guitar (E2 to E4) with 24 frets (up to E6)
register_member("guitar", lambda tick_duration, total_ticks, ticks_per_note, **kwargs: MonophonicMember(
    name="guitar",
    instrument=get_instrument("guitar"),
    tuning_system=get_tuning_system("12-TET"),
    instrument_range=[40, 88],
    tick_duration=tick_duration,
    total_ticks=total_ticks,
    ticks_per_note=ticks_per_note,
    **kwargs
))

# A 4 string bass (E1 to G3) limited to 12 frets (up to G4). This should keep the optimizer from trying to use the bass too high
register_member("bass", lambda tick_duration, total_ticks, ticks_per_note, **kwargs: MonophonicMember(
    name="bass",
    instrument=get_instrument("bass"),
    tuning_system=get_tuning_system("12-TET"),
    instrument_range=[28, 67],
    tick_duration=tick_duration,
    total_ticks=total_ticks,
    ticks_per_note=ticks_per_note,
    **kwargs
))

# A 128 key synth that supports all MIDI notes
register_member("synth", lambda tick_duration, total_ticks, ticks_per_note, **kwargs: PolyphonicMember(
    name="synth",
    instrument=get_instrument("synth"),
    tuning_system=get_tuning_system("12-TET"),
    instrument_range=[0, 127],
    tick_duration=tick_duration,
    total_ticks=total_ticks,
    ticks_per_note=ticks_per_note,
    **kwargs
))

# An 88 key synth that only plays one note at a time
register_member("synth_lead", lambda tick_duration, total_ticks, ticks_per_note, **kwargs: MonophonicMember(
    name="synth_lead",
    instrument=get_instrument("synth"),
    tuning_system=get_tuning_system("12-TET"),
    instrument_range=[21, 108],
    tick_duration=tick_duration,
    total_ticks=total_ticks,
    ticks_per_note=ticks_per_note,
    **kwargs
))
