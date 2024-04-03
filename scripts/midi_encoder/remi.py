from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from miditok import REMI, TokenizerConfig
from symusic.core import ScoreTick
from torch import Tensor

from scripts.base.base_midi_encoder import BaseMiDiEncoder


class REMI_encoder(BaseMiDiEncoder):
    def __init__(self, train_bpe=False) -> None:
        super().__init__(train_bpe)
        config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
        self.tokenizer = REMI(config)
        self.bos_token_id = self.tokenizer["BOS_None"],
        self.eos_token_id = self.tokenizer["EOS_None"],
        self.pad_token_id = self.tokenizer["PAD_None"]

    def __len__(self):
        return len(self.tokenizer.vocab)

    def encode(self, midi: ScoreTick) -> Tensor:
        """Encodes midi into ids."""
        tokens = self.tokenizer(midi)
        return torch.tensor(tokens.ids)

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]) -> ScoreTick:
        """Decodes ids and saves midi."""
        return self.tokenizer(vector)
    
    def ids_to_events(self, ids: Union[Tensor, np.ndarray, List[int]]):
        events = []
        for id in ids:
            events.append(self.tokenizer[id])
        return events
    
    def events_to_ids(self, events: List[str]):
        ids = []
        for event in events:
            ids.append(self.tokenizer[event])
        return ids

    def learn_bpe(self, files_paths: List[str], save_path: Union[str, Path], vocab_size: int=30000):
        self.tokenizer.learn_bpe(vocab_size=30000, files_paths=files_paths)
        self.tokenizer.save_params(save_path)

    def load_params(self, model_path: Union[str, Path]):
        self.tokenizer.from_pretrained(model_path)