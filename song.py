from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from members import Member
    from audio_service import AudioHandler
    # TODO
    #from loss_handler import LossHandler
    #from optim_handler import OptimHandler

class Song:
    def __init__(
        self,
        measures: int,
        tempo: int,
        beats_per_measure: int,
        ticks_per_beat: int,
        sample_rate: int
    ):
        self.members = []
        self.loss_handler = None
        self.optim_handler = None
        self.audio_handler = None

        self.measures = measures
        self.tempo = tempo
        self.beats_per_measure = beats_per_measure
        self.ticks_per_beat = ticks_per_beat
        self.sample_rate = sample_rate
    
    # Helpers for commonly used values
    def beat_duration(self) -> float:
        return 60.0 / self.tempo
    
    def tick_duration(self) -> float:
        return self.beat_duration() / self.ticks_per_beat
    
    def total_beats(self) -> int:
        return self.measures * self.beats_per_measure
    
    def total_ticks(self) -> int:
        return self.total_beats() * self.ticks_per_beat
    
    def song_duration(self) -> float:
        return self.total_ticks() * self.tick_duration()
    
    # TODO: Add serialization, save(), and load().
