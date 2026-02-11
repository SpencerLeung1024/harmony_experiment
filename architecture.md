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

The above are deferred and not assigned until after init. A Song must be initialized with:
- measures: int, number of measures
- tempo: float, beats per minute
- beats_per_measure: int, beats per measure
- ticks_per_beat: int, number of ticks per beat. Ticks are the smallest unit of time in this song. All members play notes with length a natural number of ticks.
- sample_rate: int, number of samples per second the audio should be rendered at

- beat_duration() -> float: 60.0 / tempo
- tick_duration() -> float: beat_duration() / ticks_per_beat
- total_beats() -> int: measures * beats_per_measure
- total_ticks() -> int: total_beats() * ticks_per_beat
- song_duration() -> float: total_ticks() * tick_duration()

TODO: Add serialization, save(), and load().

### Member
A Member is a band member. It can be a PolyphonicMember or a MonophonicMember.
A Member has:
- name: str, a human readable name, like "Synth" or "🎹 Tenma Saki". Will be shown in the Gradio UI
- instrument: an Instrument.
- tuning_system: a TuningSystem. Different Members of a Song can use different tuning systems
- instrument_range: list of two ints, weights.shape[0] - 1 apart. Maps weights[0, :] and weights[-1, :] to whatever TuningSystem uses
- velocity: float, for PolyphonicMembers is the ideal sum for loss calculations (they can render audio with higher or lower velocity sum), for MonophonicMembers is the actual velocity of the one note they play
- tick_duration: float, from Song.tick_duration() after song is created
- total_ticks: int, from Song.total_ticks()
- ticks_per_note: int, duration of one note from this member in terms of ticks of the song. This project does not support one member playing notes of different lengths.
- hp: a dict of string: any, hyperparameters for loss and optimization. Which hyperparameters exist depends on the exact member
- - TODO: "hp: dict for hyperparameters is flexible but error-prone; consider a typed MemberHyperparameters base class"
- weights: torch.Tensor, has shape (keys, note_times)
- painted_weights: torch.Tensor, represents user-provided constraints. Has the same shape as weights. 0.0 is interpreted as pass-through.
- painted_mask: torch.Tensor, mask of the above. Used in init for register_post_accumulate_grad_hook

Weights can be passed in as initial_weights during init.

- note_duration() -> float: Song.tick_duration() * ticks_per_note
- total_notes() -> int: Song.total_ticks() / ticks_per_note
- get_effective_weights() -> torch.Tensor: Overlays the user's painted note constraints onto the member's weights and presents a combined tensor.
- forward(x: Any) -> torch.Tensor: Returns the activations using effective weights.
- paint_weights(newly_painted_weights: torch.Tensor): For pixels that are not 0.0, paints those weights onto painted_weights, updates painted_mask, and updates weights

weights.shape[1] is total_notes().
This means Song.total_ticks() must a common multiple of all Member.ticks_per_note.

If Song.beats_per_measure = 4 (4/4 time), you want the pattern to repeat every beat, and you have:
- Member 1 playing 8th notes (8 notes per measure, 2 notes per beat)
- Member 2 playing 12th notes (12 notes per measure, 3 notes per beat)
Then Song.ticks_per_beat must be at least 6. (You can do 12, 18, etc. but without other instruments that's superfluous.)
At Song.ticks_per_beat = 6:
- Member1.ticks_per_note = 3
- Member2.ticks_per_note = 2

Note: *Only* ticks matter for timing. The code does not care about measures or beats. They serve only to present the song's structure in ways the user can interpret.
You can, as a contrived example, have:
- Song: measures=6, beats_per_measure=3, ticks_per_beat=5 -> total_ticks() = 90, ticks per measure = 15
- Member1.ticks_per_note = 9
- Member2.ticks_per_note = 10
This is basically a 10:9 polyrhythm. It is completely nonsensical since measures and beats are meaningless but the code will accept it.

### PolyphonicMember(Member)
A PolyphonicMember represents a band member that can play zero to all notes at zero to unbounded amplitude every note_time.
Piano, synth, etc.
Its weights[key, note_time] is the amplitude of key at note_time.
It uses ReLU. Due to problems with sigmoid dying we will not be using sigmoid.

### MonophonicMember(Member)
A MonophonicMember represents a band member that must play one note at a given amplitude every note_time.
Lead guitar, bass, etc.
Its weights[key, note_time] is the logit of playing key at note_time.
It uses Gumbel-softmax.
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
keys.shape = (128)
keys[0] = C-1 (8.1758 Hz)
keys[127] = G9 (12544 Hz)

The exact implementation and meaning of keys, key_to_freq(), and freq_to_key() will differ between tuning systems.
The least rigid tuning system is arbitrary frequencies provided by the user.
[100 Hz, 110 Hz, 120 Hz, 130 Hz, ...] etc.
freq_to_key() may return None for the vast majority of audible (20 Hz - 20 kHz) frequencies.

Do not expect to be able to make a meaningful comparison between raw ints of different TuningSystems.
60 may not be C4 (261.63 Hz). 69 may not be A4 (440.00 Hz).
Always use key_to_freq() and generate the dissonance matrices between members using frequencies in Hz.

### Instrument
An Instrument stores harmonics, amplitudes, and ADSR parameters.
An Instrument has:
- harmonics: a list of (multiple of the fundamental, amplitude)
- adsr: an ADSR
- harmonic_adsrs: a dict of int: list of (multiple of the fundamental, amplitude). Used to provide per-harmonic envelope overrides

- get_sound(freq: float, velocity: float, duration: float, sample_rate: int) -> np.ndarray: Returns a (samples) ndarray where each value is the contribution this instrument makes to the sound. After np.sin.
- mean_amplitudes(duration: float, sample_rate: int) -> list of (multiple of the fundamental, amplitude). Uses ADSR.mean_amplitude() to provide mean amplitude for each harmonic.

### ADSR
An ADSR stores attack, decay, sustain, and release parameters and generates an envelope.
An ADSR has:
- attack: float, seconds between key on and max amplitude
- decay: float, seconds between max amplitude and sustained amplitude
- sustain: float, sustained amplitude after decay
- release: float, seconds between key off and silence

- get_envelope(duration: float, sample_rate: float) -> np.ndarray: Returns a (samples) ndarray where each value is the amplitude at that sample.
- mean_amplitude(duration: float, sample_rate: float) -> float: get_envelope(duration).mean()

### LossHandler
A LossHandler calculates loss on a song.
A LossHandler has:
- dissonance_matrices: a dict of (int, int): torch.Tensor. (source member index, destination member index): dissonance matrix

- sample_like(dst_weights: torch.Tensor, src: Member, note_shift: int) -> torch.Tensor

note_shift = -1 means sample src.ticks_per_note from the past.
note_shift = +1 means sample src.ticks_per_note from the future (not currently used).

A dissonance matrix is the unit time, unit amplitude dissonance between two frequencies.
The actual dissonance value scales with both time (seconds) and amplitude (as a ratio to 1).

For example, suppose we had:
- 0. a piano with 88 keys: [0, 87] -> [21, 108] (A0 - C8)
- 1. a guitar with 49 "keys" (25 between the low and high E inclusive plus 24 frets): [0, 48] -> [40, 88] (E2 - E6)

dissonance_matrices[(0, 0)] is a (88, 88) matrix for dissonance to piano caused by piano
dissonance_matrices[(0, 1)] is a (88, 49) matrix for dissonance to piano caused by guitar
dissonance_matrices[(1, 0)] is a (49, 88) matrix for dissonance to guitar caused by piano
dissonance_matrices[(1, 1)] is a (49, 49) matrix for dissonance to guitar caused by guitar (note that because lead guitars here are monophonic instruments they cannot cause concurrent dissonance to themselves, only temporal dissonance)

These dissonance matrices are used to calculate:
Sum over src.total_notes()
- self_concurrent_loss for polyphonic members
- - torch.sum(src.weights.T @ D @ src.weights)
- mate_concurrent_loss
- - torch.sum(src.weights.T @ D @ sample_like(dst.weights, src, 0))

Sum over src.total_notes() - 1
- self_temporal_loss (dissonance to self note_time caused by self note_time-1)
- - torch.sum(src.weights[:, 1:].T @ D @ src.weights[:, :-1])
- mate_temporal_loss (dissonance to self note_time caused by mate note_time-1)
- - torch.sum(src.weights[:, 1:].T @ D @ sample_like(dst.weights, src, -1))

sample_like is a very tedious function that produces a tensor with shape (dst.weights.shape[0] (dst keys), src.weights.shape[1] (src notes) - abs(note_shift))
Energy-preserving horizontal scaling and cropping
Stretching to 2x halves all amplitudes. Squashing to 0.5x doubles all amplitudes. torch.sum() remains the same.
Yes I know 2x amplitude means 4x energy in physical terms but whatever
- dst.weights (dst keys, dst notes) are stretched out by a factor of dst.ticks_per_note to become temp (dst keys, Song.total_ticks())
- temp (dst keys, Song.total_ticks()) has note_shift * src.ticks_per_note cropped from the left (+) or right (-) to become temp (dst keys, Song.total_ticks() - abs(note_shift * src.ticks_per_note))
- temp is squashed in by a factor of src.ticks_per_note to become result (dst keys, src notes - abs(note_shift))
- The result has energy of each dst note as viewed by each of src's notes in its own duration.

Before the first optimization step, the actual frequencies of, for example:
Piano's key 12 and Guitar's key 34 is used in the dissonance function to fill in:
- dissonance_matrices[(0, 1)][12, 34]
- dissonance_matrices[(1, 0)][34, 12]

There are probably smart ways to reduce duplicated work but we'll get to that when we get there

Besides these expensive dissonance sandwiches, a LossHandler also calculates auxiliary losses such as:
- PolyphonicMember:
- - velocity_loss (general L1 with Member.velocity subtracted, has two different loss factors: velocity_below and velocity_above because Adam keeps killing the weights)
- - (currently disabled) quietness_loss (obviously you can have zero dissonance by not playing any notes, so encourage some activity)
- - (currently disabled) muddyness_loss (penalize small amplitudes across large numbers of keys and encourage sparse but strong notes)
- - - Currently disabled because I can't think of a good function that does what I want
- - hand_stretch_loss (play notes close together on the keyboard / close to their median)

- MonophonicMember:
- - jump_loss (don't make large jumps between notes)

- All:
- - extreme_range_loss (prefer notes near the middle of each member's range)

### OptimHandler
An OptimHandler uses the loss to optimize the members' weights.
An OptimHandler has:
- steps: int, number of steps to run

- do_steps(desired_steps: int): accepts 1, 5, 10, etc. steps from the Gradio UI. Does that many steps before pausing. This allows the user to plot the weights and loss and listen to the song as it's being cooked.

After the members are created and before the optimization loop runs, it explores the various members and their hyperparameters. Each member's weight is added to an optimizer with the specified hyperparameters.

Entirely designed and implemented by Kimi K2.5. It seems to be a typical Adam with weight clipping.

"Pixels" of user painted weights will not be optimized. A user may disable optimization on a member entirely. For example, if they manually wrote a chord progression on 🎹 Tenma Saki and just want 🎸 Hoshino Ichika and 🍜 Hinomori Shiho to make something up on top of it.
The optimizer does not need to be aware of this. Each member's forward pass already uses its effective weights, including any note constraints.

TODO: consider how to display grad state in the Gradio UI

## Services
These just exist. Their methods have @staticmethod.

### AudioService
Turns the members of the song into audio.
The AudioService has:
- def apply_note(audio: np.ndarray, sample_rate: int, instrument: Instrument, start_sample: int, freq: float, velocity: float: note_duration: float): Applies a note to the audio ndarray.
- render_member(song_duration: float, sample_rate: int, tick_duration: float, member: Member) -> np.ndarray: Returns a (samples) ndarray where each value is the contribution this member makes to the song's audio. Loops over each weight in the member's weights.
- render(song: Song) -> np.ndarray: Returns a (samples) ndarray of every member added together.
- apply_limiter(audio: np.ndarray, headroom_db: float = -6.0, peak_value: float = 0.95) -> np.ndarray: Returns an ndarray after compression. Values below headroom_db are passed through and values above go through np.tanh so they asymptotically approach peak_value.

Don't forget to use apply_limiter() after render()!

### ColorService
Gives color to a weights, activation, or spectrogram graymap.
The ColorService has:
- color_weights(member: Member) -> np.ndarray: Turns the member's weights (torch.Tensor) (num_keys, num_notes) (-inf, inf) into an (np.ndarray) (num_keys, num_notes, 3) [0.0, 1.0].
- color_activations(member: Member) -> np.ndarray: Does the same thing but after Member.forward(). Shows argmax for MonophonicMember.
- color_spectrogram(spectrogram: np.ndarray, sample_rate: int) -> np.ndarray: Turns (num_bins, num_times) [0, inf) into a (num_bins, num_times, 3) [0.0, 1.0]

I suppose if this project ever gets a large following I'll add more visualization options (different color schemes, one scheme from 20 Hz to 20 kHz instead of repeating every 2x, etc.).
But for now, the visualization I like is to map the chromatic circle to the hue circle. This gives a visual indication of the 2x scale (oh, every red is a doubling) that does not depend on the underlying tuning system (what is key 47 in 19-EDO?).
C in every octave is red (1.0, 0.0, 0.0). Hue increases with pitch class (C# is orange, etc.). Other tuning systems won't have 12 notes and 12 hues, but will still go around the chromatic circle with 19, 24, 31, or non-repeating colors.
In non-octave scales, color_weights may have non-repeating colors.
Since color_spectrogram takes a spectrogram generated from audio that came from every member, all of whom may have different tunings, it is not sensible to "round" to any particular pitch class. color_spectrogram will always assign a color to each bin, which will be slightly different from bin-1 and bin+1.
