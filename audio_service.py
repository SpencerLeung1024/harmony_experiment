import numpy as np
import torch

from song import Song
from members import Member, PolyphonicMember, MonophonicMember
from instruments import Instrument

class AudioService:
    # Helper function for render_member
    @staticmethod
    def apply_note(
        audio: np.ndarray,
        sample_rate: int,
        instrument: Instrument,
        start_sample: int,
        freq: float,
        velocity: float,
        note_duration: float,
    ):
        note_audio = instrument.get_sound(freq, velocity, note_duration, sample_rate)
        # Trim note if it goes past the end of the song
        if start_sample + len(note_audio) > len(audio):
            note_audio = note_audio[:len(audio) - start_sample]
        audio[start_sample:start_sample+len(note_audio)] += note_audio
    
    @staticmethod
    def render_member(
        song_duration: float,
        sample_rate: int,
        tick_duration: float,
        member: Member
    ) -> np.ndarray:
        samples = song_duration * sample_rate
        audio = np.zeros(int(samples))

        note_duration = member.note_duration()
        activation = member.forward(None)

        for note in range(member.num_notes):
            start_tick = note * member.ticks_per_note
            start_sample = int(start_tick * tick_duration * sample_rate)

            freq = None
            velocity = 0.0

            # This is a PolyphonicMember and weights represent amplitudes
            if isinstance(member, PolyphonicMember):
                for key in range(member.num_keys):
                    actual_key = key + member.instrument_range[0] # Convert local key index to key in the tuning system
                    freq = member.tuning_system.key_to_freq(actual_key)
                    velocity = activation[key, note].item()
                    if freq and velocity > 0.0:
                        AudioService.apply_note(audio, sample_rate, member.instrument, start_sample, freq, velocity, note_duration)
                    
            # This is a MonophonicMember and weights represent probabilities
            elif isinstance(member, MonophonicMember):
                # When rendering, use argmax
                key = torch.argmax(activation[:, note]).item()
                actual_key = key + member.instrument_range[0]
                freq = member.tuning_system.key_to_freq(actual_key)
                velocity = 0.5 # Ideally this should be a property of a MonophonicMember but for now just make all of them play at 50%
                if freq and velocity > 0.0:
                    AudioService.apply_note(audio, sample_rate, member.instrument, start_sample, freq, velocity, note_duration)
        
        return audio
    
    @staticmethod
    def render(
        song: Song
    ) -> np.ndarray:
        # Render each member and sum them together
        audio = np.zeros(int(song.song_duration() * song.sample_rate))
        for member in song.members:
            member_audio = AudioService.render_member(
                song.song_duration(),
                song.sample_rate,
                song.tick_duration(),
                member
            )
            audio[:len(member_audio)] += member_audio
        return audio
