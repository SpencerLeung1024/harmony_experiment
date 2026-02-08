from typing import List

from members import Member
# TODO
#from loss_handler import LossHandler
#from optim_handler import OptimHandler
#from audio_handler import AudioHandler

class Song:
    def __init__(
        self,
        members: List[Member],
        #loss_handler: LossHandler,
        #optim_handler: OptimHandler,
        #audio_handler: AudioHandler,
        measures: int,
        tempo: int,
        beats_per_measure: int,
        ticks_per_beat: int
    ):
        self.members = members
        #self.loss_handler = loss_handler
        #self.optim_handler = optim_handler
        #self.audio_handler = audio_handler
        self.measures = measures
        self.tempo = tempo
        self.beats_per_measure = beats_per_measure
        self.ticks_per_beat = ticks_per_beat
    
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
