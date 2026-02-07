# Harmony Experiment Architecture

This one is typed by a human.
Kimi K2.5's hands-off implementation of all of Future Plans is cool but isn't exactly what I wanted. It can serve as reference for things though.

## Objects

### Song
A Song is the largest object. It encompasses all settings the user put in the Gradio UI and is sent to the backend.
A Song has:
- members: list of 0 or more Members
- loss_handler: a LossHandler
- optim_handler: an OptimHandler
- audio_handler: an AudioHandler
- measures: int, number of measures
- bpm: float, beats per minute
- beats_per_measure: int, beats per measure
- subdivisions: int, number of subdivisions per beat

- beat_time() -> float: bpm / 60
- subdivision_time() -> float: beat_time() / subdivisions
- total_beats() -> int: measures * beats_per_measure
- total_subdivisions() -> int: total_beats() * subdivisions
- total_time() -> float: total_subdivisions() * subdivision_time()

### Member
A Member is a band member. It can be a PolyphonicMember or a MonophonicMember.
A Member has:
- name: str, a human readable name, like "Synth" or "🎹 Tenma Saki". Will be shown in the Gradio UI
- weights: torch.Tensor, has shape (keys, time_thingys)
- tuning_system: a TuningSystem. Different Members of a Song can use different tuning systems
- instrument_range: list of two ints, weights.shape[0] - 1 apart. Maps weights[0, :] and weights[-1, :] to whatever TuningSystem uses
- instrument: an Instrument.
- hp: a dict of string: any, hyperparameters for loss and optimization. Which hyperparameters exist depends on the exact member
- time_thingy_size: int, number of subdivisions that one time_thingy represents

- time_thingy_time() -> float: Song.subdivision_time() * time_thingy_size
- total_time_thingys() -> int: Song.total_subdivisions() / time_thingy_size

weights.shape[1] is total_time_thingys().
This means:
- Song subdivisions must be the lowest common multiple of all members. If you have Member 1 playing 8th notes and Member 2 playing 12th notes, the song must have a multiple of (8, 12) = 24 subdivisions. Member 1 would have time_thingy_size = 3 and Member 2 would have time_thingy_size = 2.

### PolyphonicMember(Member)
A PolyphonicMember represents a band member that can play zero to all notes at zero to unbounded amplitude every time_thingy.
Piano, synth, etc.
Its weights[key, time_thingy] is the amplitude of key at time_thingy.
It uses ReLU. Due to problems with sigmoid dying we will not be using sigmoid.

### MonophonicMember(Member)
A MonophonicMember represents a band member that must play one note at a given amplitude every time_thingy.
Lead guitar, bass, etc.
Its weights[key, time_thingy] is the logit of playing key at time_thingy.
It uses Gumbel-Softmax.
It uses a straight-through estimator:
- forward: argmax
- backward: Gumbel-softmax

### TuningSystem
A TuningSystem is the frequencies that a member's weights.shape[0] (keys) refers to.
It is not necessarily 12-TET, A4=440Hz. That standard is just first among equals.
Different members may use different tuning systems. You can have a song with two pianos, tuned a quarter tone apart, to see what kind of avant-garde pieces the optimizer finds.
- keys: np.ndarray: A 1D vector of all keys that exist in the tuning system, with frequencies in Hz.

- key_to_freq(key: int) -> float: Usually keys[key]. For tunings in which this can be generated programatically, this function may support keys outside of the ndarray above.
! Can return None.
- freq_to_key(freq: float) -> int: Usually programatically rounds to the nearest key.
! Can return None.

For example, in 12-TET, A4=440Hz:
keys.shape[0] = 128
keys[0] = C-1 (8.1758 Hz)
keys[127] = G9 (12544 Hz)

Do not expect to be able to make a meaningful comparison between raw ints of different TuningSystems.
60 may not be C4 (261.63 Hz). 69 may not be A4 (440.00 Hz).
Always use key_to_freq() and generate the dissonance matrices between members using frequencies in Hz.

### Instrument
An Instrument stores harmonics, amplitudes, and ADSR parameters.
An Instrument has:
- harmonics: a list of (multiple of the fundamental, amplitude)
- adsr: an ADSR
- harmonic_adsrs: a dict of int: list of (multiple of the fundamental, amplitude). Used to provide per-harmonic envelope overrides

- get_sound(duration: float, velocity: float) -> np.ndarray: Returns a (samples) ndarray where each value is the contribution this instrument makes to the sound. After np.sin.
- mean_amplitudes(duration: float) -> list of (multiple of the fundamental, amplitude). Uses ADSR.mean_amplitude to provide mean amplitude for each harmonic.

### ADSR
An ADSR stores attack, decay, sustain, and release parameters and generates an envelope.
An ADSR has:
- attack: float, seconds between key on and max amplitude
- decay: float, seconds between max amplitude and sustained amplitude
- sustain: float, sustained amplitude after decay
- release: float, seconds between key off and silence

- get_envelope(duration: float) -> np.ndarray: Returns a (samples) ndarray where each value is the amplitude at that sample.
- mean_amplitude(duration: float) -> float: get_envelope(duration).mean()

### LossHandler
A LossHandler calculates loss on a song.
A LossHandler has:
- dissonance_matrices: a dict of (int, int): torch.Tensor. (source member index, destination member index): dissonance matrix

A dissonance matrix is the unit time, unit amplitude dissonance between two frequencies.
The actual dissonance value scales with both time (seconds) and amplitude (as a ratio to 1).

For example, suppose we had:
- 0. a piano with 88 keys: [0, 87] -> [21, 108] (A0 - C8)
- 1. a guitar with 49 "keys" (25 between the low and high E inclusive plus 24 frets): [0, 48] -> [40, 88] (E2 - E6)

dissonance_matrices[(0, 0)] is a (88, 88) matrix for dissonance to piano caused by piano
dissonance_matrices[(0, 1)] is a (88, 49) matrix for dissonance to piano caused by guitar
dissonance_matrices[(1, 0)] is a (49, 88) matrix for dissonance to guitar caused by piano
dissonance_matrices[(1, 1)] is a (49, 49) matrix for dissonance to guitar caused by guitar (note that because lead guitars here are monophonic instruments they cannot cause concurrent dissonance to themselves, only temporal dissonance)

Before the first optimization step, the actual frequencies of, for example:
Piano's key 12 and Guitar's key 34 is used in the dissonance function to fill in:
- dissonance_matrices[(0, 1)][12, 34]
- dissonance_matrices[(1, 0)][34, 12]

There are probably smart ways to reduce duplicated work but we'll get to that when we get there

loss = loss + torch.sum(src.weights.T @ D @ dst.weights)

### OptimHandler
An OptimHandler uses the loss to optimize the members' weights.
An OptimHandler has:
- steps: int, number of steps to run

After the members are created and before the optimization loop runs, it explores the various members and their hyperparameters. Each member's weight is added to an optimizer with the specified hyperparameters.

"Pixels" of user painted weights will not be optimized. A user may disable optimization on a member entirely. For example, if they manually wrote a chord progression on a synth and just want 🎸 Hoshino Ichika and 🍜 Hinomori Shiho to make something up on top of it.

### AudioHandler
Turns the members of the song into audio.
An AudioHandler has:
- sample_rate: int, number of samples per second

### ColorService
Gives color to a weights graymap or a spectrogram graymap.
It is not an object. It just exists.
The ColorService has:
- color_weights(weights: np.ndarray, tuning_system: TuningSystem) -> np.ndarray: Turns (num_keys, num_time_thingys) (-inf, inf) into a (num_keys, num_time_thingys, 3) [0.0, 1.0]
- color_spectrogram(spectrogram: np.ndarray, sample_rate: int) -> np.ndarray: Turns (num_bins, num_times) [0, inf) into a (num_bins, num_times, 3) [0.0, 1.0]

C in every octave is red (1.0, 0.0, 0.0). Hue increases with pitch class (C# is orange, etc.).
In non-octave scales, color_weights may have non-repeating colors.
Since color_spectrogram takes a spectrogram generated from audio that came from every member, all of whom may have different tunings, it is not sensible to "round" to any particular pitch class. color_spectrogram will always show the color of the bin.
