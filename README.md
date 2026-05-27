# Harmony From First Principles

An experiment to optimize instruments in a tuning system into chords and chord loops.

I've been sitting on this for too long. This is from February 2026.

Publishing to Github so at least it will exist in the world.

## Rationale

- Music is a subset of sound. There is significant disagreement about which subset, but depending on how you construct your code assumptions and loss functions you can choose which subset to enforce.
- All sound can be expressed as sums of sine waves.
- The perceived dissonance between two sine waves depends on ratio, with a peak around 1.03 (50 cents).
- By minimizing dissonance between harmonics of different notes, we can optimize for a chord.

## Features

- Variable number of measures, beats, and ticks
- Variable members with Matplotlib adjusting to accomodate
- Hue-based visualization for comparisons between tuning systems and octaves
- 20 instruments using additive synthesis
- -  Default has 🎹 piano, 🎸 guitar, 🍜 bass, and (partial) 🥁 drums
- -  Choir vowels with formant shaping
- Hyperparameters for auxiliary losses to shape dissonance weight and playing mechanics
- Interactive and CLI mode (see `main.py`)

## Examples

![Activations at Step 100](IMAGES/activations_step0100.png)

![Spectrogram at Step 100](IMAGES/spectrogram_step0100.png)

![Audio at Step 100](audio_step0100.wav)

## Acknowledgements

I used Kimi K2.5 for advice on how to turn music into a PyTorch and NumPy problem. `DERIVATION.md`, `COMPRESSION.md`, `POLYPHONICITY.md`, and `VOICE.md`, as well as instrument parameters, were provided by the model.

Professor Stefan Smulovitz
- CA 149: Sound, 2024 Summer Term, SFU
- https://www.sfu.ca/outlines.html?2024/summer/ca/149/ol01
- You taught me to stop and think about sound, to break it down into its components, and that music is not as simple of a category as the 12 notes of Western music.

minutephysics - The Physics of Dissonance
- https://www.youtube.com/watch?v=tCsl6ZcY9ag
- The specific method of summing dissonance between harmonics came from this video.
