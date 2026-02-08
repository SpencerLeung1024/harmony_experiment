from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
import torch

from song import Song
from instruments import Instrument
from tuning_systems import TuningSystem

class Member(ABC):
    def __init__(
        self,
        song: Song,
        name: str,
        instrument: Instrument,
        tuning_system: TuningSystem,
        instrument_range: List[int],
        num_notes: int,
        ticks_per_note: int,
        hp: Dict[str, Any],
        initial_weights: Optional[torch.Tensor] = None
    ):
        self.song = song
        self.name = name
        self.tuning_system = tuning_system
        self.instrument_range = instrument_range
        self.hp = hp
        self.ticks_per_note = ticks_per_note

        # Check if instrument_range is two ints, then calculate num_keys
        if len(instrument_range) != 2:
            raise ValueError(f"instrument_range must be a list of two ints. Got {instrument_range}.")
        num_keys = instrument_range[1] - instrument_range[0] + 1
        self.num_keys = num_keys
        self.num_notes = num_notes

        # Check if song.total_ticks() is a multiple of ticks_per_note
        if self.song.total_ticks() % self.ticks_per_note != 0:
            raise ValueError(f"song.total_ticks() must be a multiple of ticks_per_note. Got {self.song.total_ticks()} total ticks and {self.ticks_per_note} ticks per note.")
        
        # Initialize or use provided weights
        if initial_weights is not None:
            if initial_weights.shape != (num_keys, num_notes):
                raise ValueError(f"initial_weights must have shape ({num_keys}, {num_notes}). Got {initial_weights.shape}."
                )
            self.weights = torch.nn.Parameter(initial_weights)
        else:
            self.initialize()
    
    # Helpers for commonly used values
    def note_duration(self) -> float:
        return self.song.tick_duration() * self.ticks_per_note
    
    def total_notes(self) -> int:
        return self.song.total_ticks() // self.ticks_per_note
    
    @abstractmethod
    def initialize(self):
        pass

class PolyphonicMember(Member):
    def __init__(
        self,
        song: Song,
        name: str,
        tuning_system: TuningSystem,
        instrument_range: List[int],
        num_keys: int,
        num_notes: int,
        ticks_per_note: int,
        hp: Dict[str, Any],
        initial_weights: Optional[torch.Tensor] = None
    ):
        super().__init__(
            song,
            name,
            tuning_system,
            instrument_range,
            num_keys,
            num_notes,
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

class MonophonicMember(Member):
    def __init__(
        self,
        song: Song,
        name: str,
        tuning_system: TuningSystem,
        instrument_range: List[int],
        num_keys: int,
        num_notes: int,
        ticks_per_note: int,
        hp: Dict[str, Any],
        initial_weights: Optional[torch.Tensor] = None
    ):
        super().__init__(
            song,
            name,
            tuning_system,
            instrument_range,
            num_keys,
            num_notes,
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
