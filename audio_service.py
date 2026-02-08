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
        print(f"Beginning rendering of member {member.name}")
        samples = song_duration * sample_rate
        audio = np.zeros(int(samples))

        note_duration = member.note_duration()

        for note in range(member.num_notes):
            start_tick = note * member.ticks_per_note
            start_sample = int(start_tick * tick_duration * sample_rate)

            freq = None
            velocity = 0.0

            # This is a PolyphonicMember and weights represent amplitudes
            print(f"What is member {member.name}?")
            if member is PolyphonicMember:
                print("PolyphonicMember")
                for key in range(member.num_keys):
                    freq = member.tuning_system.key_to_freq(key)
                    velocity = member.weights[key, note].item()
                    print(f"Member {member.name}, note {note}, key {key}: freq={freq:.2f}Hz, velocity={velocity:.4f}")
                    if freq and velocity > 0.0:
                        print(f"Applying note for member {member.name}: key={key}, freq={freq:.2f}Hz, velocity={velocity:.4f}, start_sample={start_sample}, note_duration={note_duration:.4f}s")
                        AudioService.apply_note(audio, sample_rate, member.instrument, start_sample, freq, velocity, note_duration)
                    
            # This is a MonophonicMember and weights represent probabilities
            elif member is MonophonicMember:
                print("MonophonicMember")
                # When rendering, use argmax
                key = torch.argmax(member.weights[:, note]).item()
                freq = member.tuning_system.key_to_freq(key)
                velocity = member.weights[key, note].item()
                print(f"Member {member.name}, note {note}: key={key}, freq={freq:.2f}Hz, velocity={velocity:.4f}")
                if freq and velocity > 0.0:
                    print(f"Applying note for member {member.name}: key={key}, freq={freq:.2f}Hz, velocity={velocity:.4f}, start_sample={start_sample}, note_duration={note_duration:.4f}s")
                    AudioService.apply_note(audio, sample_rate, member.instrument, start_sample, freq, velocity, note_duration)
            
            else:
                print("Unknown member type")

        
        print(audio.shape, audio.min(), audio.max(), audio.mean())
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
