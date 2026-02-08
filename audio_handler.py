import numpy as np
import torch

from song import Song
from members import Member, PolyphonicMember, MonophonicMember
from instruments import Instrument

class AudioHandler:
    def __init__(
        self,
        song: Song,
        sample_rate: int,
    ):
        self.song = song
        self.sample_rate = sample_rate
    
    # Helper function for render_member
    def apply_note(
        self,
        audio: np.ndarray,
        instrument: Instrument,
        start_sample: int,
        freq: float,
        velocity: float,
        note_duration: float,
    ):
        note_audio = instrument.get_sound(freq, velocity, note_duration, self.sample_rate)
        # Trim note if it goes past the end of the song
        if start_sample + len(note_audio) > len(audio):
            note_audio = note_audio[:len(audio) - start_sample]
        audio[start_sample:start_sample+len(note_audio)] += note_audio
    
    def render_member(
        self,
        member: Member
    ) -> np.ndarray:
        samples = self.song.song_duration() * self.sample_rate
        audio = np.zeros(int(samples))

        note_duration = member.note_duration()

        for note in range(member.num_notes):
            start_tick = note * member.ticks_per_note
            start_sample = int(start_tick * self.song.tick_duration() * self.sample_rate)

            freq = None
            velocity = 0.0

            # This is a PolyphonicMember and weights represent amplitudes
            if member is PolyphonicMember:
                for key in range(member.num_keys):
                    freq = member.tuning_system.key_to_freq(key)
                    velocity = member.weights[key, note].item()
                    if freq and velocity > 0.0:
                        self.apply_note(audio, member.instrument, start_sample, freq, velocity, note_duration)
                    
            # This is a MonophonicMember and weights represent probabilities
            elif member is MonophonicMember:
                # When rendering, use argmax
                key = torch.argmax(member.weights[:, note]).item()
                freq = member.tuning_system.key_to_freq(key)
                velocity = member.weights[key, note].item()
                if freq and velocity > 0.0:
                    self.apply_note(audio, member.instrument, start_sample, freq, velocity, note_duration)

        return audio
    
    def render(
        self
    ) -> np.ndarray:
        # Render each member and sum them together
        audio = np.zeros(int(self.song.song_duration() * self.sample_rate))
        for member in self.song.members:
            member_audio = self.render_member(member)
            audio[:len(member_audio)] += member_audio
        return audio
