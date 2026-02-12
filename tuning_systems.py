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
    for this_octave in range(-1, 10): # C-1 to G9
        # Apply the 12 ratios to this octave
        for pitch_class in range(12):
            this_key = ((this_octave+1) * 12) + pitch_class
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
# I have arbitrarily used C-1 = 8.1758 Hz as the min frequency, and G9 = 12544 Hz as the max frequency

# EDO system

MIDI_FREQ_LOW = 8.1757 # The full precision of C-1 is 8.175798915643707 Hz but 8.1758 will exclude the actual C-1, so we need to round down to expand our range
MIDI_FREQ_HIGH = 12544 # The full precision of G9 is 12543.853951415975 Hz

# This does not support reference_key since it is no longer compatible with MIDI
# It still has a reference_freq though. The keys extend downwards to freq_low and upwards to freq_high from reference_freq
def step_ratio_to_keys(step_ratio: float, divisions: int, reference_freq: float, freq_low: float, freq_high: float) -> np.ndarray:
    # Helper function
    # Returns a negative value if f2 is below f1, and a positive value if f2 is above f1
    def get_float_steps_to(f1: float, f2: float) -> float:
        return divisions * (np.log(f2 / f1) / np.log(step_ratio))
    
    # Figure out how many steps to the key_low and key_high such that they remain in freq_low and freq_high
    steps_to_low = int(np.ceil(get_float_steps_to(reference_freq, freq_low))) # Will be negative like -15.6, so we round up to -15
    steps_to_high = int(np.floor(get_float_steps_to(reference_freq, freq_high))) # Will be positive like 23.3, so we round down to 23

    # Helper function
    def get_ratio_for_steps(steps: int) -> float:
        return step_ratio ** (steps / divisions)
    
    num_keys = steps_to_high - steps_to_low + 1 # Gotta include the reference key itself
    keys = np.zeros(num_keys)
    for key in range(num_keys):
        steps_from_reference = key + steps_to_low
        keys[key] = reference_freq * get_ratio_for_steps(steps_from_reference)

    return keys

# All of these use "A4"=440 Hz, in line with the v2 implementation
# Personally I prefer my octaves starting at C but the full precision of C4 is 261.6255653005986 Hz which is awful to work with
# (it no longer makes sense to talk about *the* A4, but it sounds like an A4)

# Near-just thirds and sixths, distinct major/minor whole tones
register_tuning_system("19-EDO", lambda: TuningSystem(
    name="19-EDO",
    keys=step_ratio_to_keys(2.0, 19, 440.0, MIDI_FREQ_LOW, MIDI_FREQ_HIGH) # 201 keys
))

# Quarter-tone system used in some contemporary Arabic music
register_tuning_system("24-EDO", lambda: TuningSystem(
    name="24-EDO",
    keys=step_ratio_to_keys(2.0, 24, 440.0, MIDI_FREQ_LOW, MIDI_FREQ_HIGH) # 255 keys
))

# Excellent approximation to meantone temperament
register_tuning_system("31-EDO", lambda: TuningSystem(
    name="31-EDO",
    keys=step_ratio_to_keys(2.0, 31, 440.0, MIDI_FREQ_LOW, MIDI_FREQ_HIGH) # 328 keys
))

# Very accurate just intonation approximations
register_tuning_system("41-EDO", lambda: TuningSystem(
    name="41-EDO",
    keys=step_ratio_to_keys(2.0, 41, 440.0, MIDI_FREQ_LOW, MIDI_FREQ_HIGH) # 434 keys
))

# Close approximation to Pythagorean and just intonation
register_tuning_system("53-EDO", lambda: TuningSystem(
    name="53-EDO",
    keys=step_ratio_to_keys(2.0, 53, 440.0, MIDI_FREQ_LOW, MIDI_FREQ_HIGH) # 561 keys
))

# For the generic EDO system below, the default parameters is equivalent to standard Western tuning (12-TET)
register_tuning_system("EDO", lambda divisions=12, reference_freq=440.0, freq_low=MIDI_FREQ_LOW, freq_high=MIDI_FREQ_HIGH: TuningSystem(
    name="EDO",
    keys=step_ratio_to_keys(2.0, divisions, reference_freq, freq_low, freq_high)
))

# Uses a non-octave step ratio

# Non-octave system

# Wendy Carlos's Alpha, Beta, and Gamma scales calculate a step size in cents that is good at representing perfect fifths, major thirds, and minor thirds
# The actual values used in Beauty in the Beast are approximately:
# Alpha: 9 = P5, 5 = M3, 4 = m3: 77.964989544 cents = 1.046063785206031 ratio
# Beta: 11 = P5, 6 = M3, 5 = m3: 63.832932576 cents = 1.037559527833274 ratio
# Gamma: 20 = P5, 11 = M3, 9 = m3: 35.0985422804 cents = 1.020480620635678 ratio
# In this code the scales are approximated only by dividing the fifth

# 701.9550008653874 cents / 9 = 77.99500009615416 cents
register_tuning_system("Alpha", lambda: TuningSystem(
    name="Alpha",
    keys=step_ratio_to_keys(3/2, 9, 440.0, MIDI_FREQ_LOW, MIDI_FREQ_HIGH) # 163 keys
))

# 701.9550008653874 cents / 11 = 63.8140909877625 cents
register_tuning_system("Beta", lambda: TuningSystem(
    name="Beta",
    keys=step_ratio_to_keys(3/2, 11, 440.0, MIDI_FREQ_LOW, MIDI_FREQ_HIGH) # 199 keys
))

# 701.9550008653874 cents / 20 = 35.09775004326937 cents
register_tuning_system("Gamma", lambda: TuningSystem(
    name="Gamma",
    keys=step_ratio_to_keys(3/2, 20, 440.0, MIDI_FREQ_LOW, MIDI_FREQ_HIGH) # 362 keys
))

# The Bohlen-Pierce scale uses the tritave (3:1) instead of the octave (2:1)
# It divides this into 13 steps
# 1901.9550008653875 cents / 13 = 146.30423083579905 cents
register_tuning_system("Bohlen-Pierce", lambda: TuningSystem(
    name="Bohlen-Pierce",
    keys=step_ratio_to_keys(3.0, 13, 440.0, MIDI_FREQ_LOW, MIDI_FREQ_HIGH) # 87 keys
))

# Despite being named "Non-Octave System", the generic non-octave system below with default parameters is equivalent to standard Western tuning (12-TET)
register_tuning_system("Non-Octave System", lambda step_ratio=2.0, divisions=12, reference_freq=440.0, freq_low=MIDI_FREQ_LOW, freq_high=MIDI_FREQ_HIGH: TuningSystem(
    name="Non-Octave System",
    keys=step_ratio_to_keys(step_ratio, divisions, reference_freq, freq_low, freq_high)
))

# Lastly, the least rigid tuning system is arbitrary frequencies provided by the user
