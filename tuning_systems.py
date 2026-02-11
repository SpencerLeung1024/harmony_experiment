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

# Uses the octave, has 12 notes per octave
# For compatibility and comparability, every tuning system in this section will use the 128 notes of MIDI.

# 12-TET, MIDI standard, A4=440Hz, A4=69, 0-127
register_tuning_system("12-TET", lambda: TuningSystem(
    name="12-TET",
    keys=np.array([440 * 2**((k-69)/12) for k in range(128)])
))

# Pythagorean tuning
# Actually a special case of meantone tuning with 0 comma
# Start with a reference key and frequency
# Stack fifths to go around the circle of fifths
# Divide or multiply by 2 to keep within one octave
# One fifth will be a "wolf fifth", the others will be perfect fifths
# Traditionally, D4 = MIDI note 62 = 293.33 Hz is used as the reference
# The wolf fifth is between G# (ref-6) and Eb or D# (ref+1)
# At the end, multiply by powers of 2 to fill out the full MIDI range

# Since we want compatibility with the 128 notes of MIDI, we need to return a (12) ndarray where [0] is C in an octave
def get_ratios(reference_key: int, comma_fraction: float) -> np.ndarray:
    # Use the comma
    syntonic_comma = 81.0 / 80.0
    tempered_fifth = (3.0/2.0) / (syntonic_comma ** comma_fraction)

    # Figure out what pitch class the reference is
    reference_pitch_class = reference_key % 12

    ratios = np.zeros(12)
    ratios[reference_pitch_class] = 1.0
    
    current_pitch_class = reference_pitch_class
    # Go up the circle of fifths
    # Using D, this should be D, A, E, B, F#, C#, G# (stop here)
    for i in range(1, 7):
        current_pitch_class = (current_pitch_class + 7) % 12
        current_ratio = tempered_fifth ** i
        # Do not enforce one octave here. We will do that at the end
        ratios[current_pitch_class] = current_ratio
    
    current_pitch_class = reference_pitch_class
    # Go down the circle of fifths
    # Using D, this should be D, G, C, F, A#, D# (stop here)
    for i in range(1, 6):
        current_pitch_class = (current_pitch_class - 7) % 12
        current_ratio = (1 / tempered_fifth) ** i
        ratios[current_pitch_class] = current_ratio
    
    # But wait! Octaves in MIDI are defined as happening every C
    # So we need to ensure ratios[0] is the lowest among ratios
    # Since we know the reference is in the octave somewhere, this C must be in (0.5, 1.0] relative to the reference
    while ratios[0] > 1.0:
        ratios[0] /= 2
    while ratios[0] <= 0.5:
        ratios[0] *= 2
    # Now, for pitch classes between C and the reference, we need to place them in (0.5, 1.0]
    for i in range(1, reference_pitch_class): # May run 0 times if reference is C
        while ratios[i] > 1.0:
            ratios[i] /= 2
        while ratios[i] <= 0.5:
            ratios[i] *= 2
    # The pitch classes above the reference must be in [1.0, 2.0)
    for i in range(reference_pitch_class+1, 12): # May run 0 times if reference is B
        while ratios[i] >= 2.0:
            ratios[i] /= 2
        while ratios[i] < 1.0:
            ratios[i] *= 2
    
    return ratios

def ratio_to_keys(ratios: np.ndarray, reference_key: int, reference_freq: float) -> np.ndarray:
    # Figure out which octave the reference is in
    octave = reference_key // 12

    keys = np.zeros(128)
    for this_octave in range(-1, 9): # C-1 to G9
        # Apply the 12 ratios to this octave
        for pitch_class in range(12):
            this_key = (this_octave * 12) + pitch_class
            if this_key > 127:
                break
            keys[this_key] = reference_freq * ratios[pitch_class] * (2 ** (this_octave - octave))
    
    return keys
        
register_tuning_system("Pythagorean", lambda: TuningSystem(
    name="Pythagorean",
    keys=ratio_to_keys(get_ratios(62, 0.0), 62, 293.33)
))

# Meantone tuning
# The v2 implementation uses C4 = MIDI note 60 = 261.626 Hz as the reference so I will keep that for v3

register_tuning_system("1/4-comma Meantone", lambda: TuningSystem(
    name="1/4-comma Meantone",
    keys=ratio_to_keys(get_ratios(60, 1/4), 60, 261.626)
))

register_tuning_system("1/3-comma Meantone", lambda: TuningSystem(
    name="1/3-comma Meantone",
    keys=ratio_to_keys(get_ratios(60, 1/3), 60, 261.626)
))

register_tuning_system("Meantone", lambda reference_key=60, reference_freq=261.626, comma_fraction=1/4: TuningSystem(
    name="Meantone",
    keys=ratio_to_keys(get_ratios(reference_key, comma_fraction), reference_key, reference_freq)
))

# Uses the octave, does not have 12 notes per octave
# These cannot work with MIDI
# I have arbitrarily used C-1 = 8.1758 Hz as note 0, and G9 = 12544 Hz as the max frequency

# EDO system

# Uses a non-octave step ratio
# Like above, C-1 = 8.1758 Hz is note 0, and G9 = 12544 Hz is the max frequency

# Non-octave system

# Lastly, the least rigid tuning system is arbitrary frequencies provided by the user
