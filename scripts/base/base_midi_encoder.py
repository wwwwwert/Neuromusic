from typing import List, Union

import numpy as np
from symusic.core import ScoreTick
from torch import Tensor


class BaseMiDiEncoder:
    def __init__(self, train_bpe=False) -> None:
        self.train_bpe = False
        
    def encode(self, midi: ScoreTick) -> Tensor:
        """Encodes midi into ids."""
        raise NotImplementedError()

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]) -> ScoreTick:
        """Decodes ids and saves midi."""
        raise NotImplementedError()
    
    def ids_to_events(self, ids: Union[Tensor, np.ndarray, List[int]]):
        raise NotImplementedError
    
    def events_to_ids(self, events: List[str]):
        raise NotImplementedError

    def learn_bpe(self, files_paths: List[str], vocab_size: int=30000):
        raise NotImplementedError()

    def save_params(self):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

    # def normalize_midi(self, tokens: np.ndarray):
    #     """Removes redundant pauses at the beginning and end of the piece."""
    #     raise NotImplementedError()
