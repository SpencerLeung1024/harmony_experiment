from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class TuningSystem(ABC):
    def __init__(
        self,
        keys: np.ndarray
    ):
        self.keys = keys
    
    def key_to_freq(self, key: int) -> Optional[float]:
        if key < 0 or key >= len(self.keys):
            return None
        return self.keys[key]
    
    @abstractmethod
    def freq_to_key(self, freq: float) -> Optional[int]:
        pass
