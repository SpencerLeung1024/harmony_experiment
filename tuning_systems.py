from typing import Optional, Union, Callable
import numpy as np

class TuningSystem:
    def __init__(
        self,
        name: str,
        keys: np.ndarray
    ):
        self.name = name
        self.keys = keys
    
    def key_to_freq(self, key: int) -> Optional[float]:
        if key < 0 or key >= len(self.keys):
            return None
        return self.keys[key]
    
    def freq_to_key(self, freq: float) -> Optional[int]:
        # Default: nearest neighbor with midpoints
        if freq < self.keys[0] * 0.95 or freq > self.keys[-1] * 1.05:
            return None
        midpoints = (self.keys[:-1] + self.keys[1:]) / 2
        idx = np.searchsorted(midpoints, freq)
        return int(idx)

# Use a registry pattern to turn str into default objects
_TUNING_SYSTEM_REGISTRY = {}

def register_tuning_system(
    name: str,
    factory: Callable[..., TuningSystem]
):
    _TUNING_SYSTEM_REGISTRY[name] = factory

def get_tuning_system(
    name_or_instance: Union[str, TuningSystem],
    **kwargs
) -> TuningSystem:
    if isinstance(name_or_instance, TuningSystem):
        return name_or_instance
    if name_or_instance not in _TUNING_SYSTEM_REGISTRY:
        raise ValueError(f"Unknown tuning system: {name_or_instance}")
    return _TUNING_SYSTEM_REGISTRY[name_or_instance](**kwargs)

# Register defaults

# MIDI standard, 12-TET, A4=440Hz, A4=69, 0-127
register_tuning_system("12-TET", lambda: TuningSystem(
    name="12-TET",
    keys=np.array([440 * 2**((k-69)/12) for k in range(128)])
))

# TODO: Add other tuning systems from v2
