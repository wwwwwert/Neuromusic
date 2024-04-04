import logging
import random
from typing import List

import numpy as np
import torch
from symusic import Score, TimeUnit
from torch import Tensor
from torch.utils.data import Dataset

from miditok.midi_tokenizer import MIDITokenizer
from scripts.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            midi_encoder: MIDITokenizer,
            config_parser: ConfigParser,
            max_items: int=-1,
            audio_length: int=-1,
            n_tokens: int=1024,
            midi_augs=None,
            tokens_augs=None, 
            max_audio_length=None,
            filter_limit=None,
            train_bpe=False,
            **kwargs
    ):
        self.midi_encoder = midi_encoder
        self.config_parser = config_parser
        self.audio_length = audio_length
        self.n_tokens = n_tokens
        self.midi_augs = midi_augs

        if train_bpe:
            midi_files = [obj['midi_path'] for obj in index]
            midi_encoder.learn_bpe(midi_files)
        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, filter_limit)
        if max_items > 0:
            index = random.sample(index, max_items)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        midi = self.load_midi(data_dict['midi_path'])
        tokens, mask = self.get_tokens(midi)
        return {
            "tokens": tokens,
            "midi_path": data_dict['midi_path'],
            "midi": midi,
            "tokens_mask": mask,
            "sequence_length": mask.sum().item(),
        }

    @staticmethod
    def _sort_index(index):
        features = list(index[0].keys())
        if "chords_density" in features:
            # number of chords per ms
            index = sorted(index, key=lambda x: x["chords_density"])
        elif "duration" in features:
            index = sorted(index, key=lambda x: x["duration"])
        return index

    def __len__(self):
        return len(self._index)

    def load_midi(self, path):
        midi = Score(path, ttype=TimeUnit.second)
        if self.audio_length != -1 and midi.end() > self.audio_length:
            random_position = random.uniform(0, midi.end() - self.audio_length)
            midi = midi.clip(random_position, random_position + self.audio_length)
            midi = midi.shift_time(-random_position)
        midi = midi.to(TimeUnit.tick)

        if self.midi_augs is not None:
            midi = self.midi_augs(midi)

        midi = midi.sort()
        return midi
    
    def get_tokens(self, midi):
        tokens_list = self.midi_encoder(midi).ids
        tokens = torch.zeros(len(tokens_list) + 2, dtype=torch.int32)
        tokens[0] = self.midi_encoder['BOS_None']
        tokens[-1] = self.midi_encoder['EOS_None']
        tokens[1:-1] = torch.tensor(tokens_list)
        mask = torch.ones(self.n_tokens, dtype=torch.int32)
        if tokens.shape[0] < self.n_tokens:
            mask[tokens.shape[0]:] = 0
            n_pad = self.n_tokens - tokens.shape[0]
            tokens = torch.nn.functional.pad(tokens, (0, n_pad), 'constant', value=self.midi_encoder["PAD_None"])
        elif tokens.shape[0] > self.n_tokens:
            random_position = random.randint(0, tokens.shape[0] - self.n_tokens)
            tokens = tokens[random_position:random_position + self.n_tokens]
        return tokens, mask
    
    def get_tokens_with_prev(self, midi):
        tokens_list = self.midi_encoder(midi).ids
        tokens = torch.zeros(len(tokens_list) + 2, dtype=torch.int32)
        tokens[0] = self.midi_encoder['BOS_None']
        tokens[-1] = self.midi_encoder['EOS_None']
        tokens[1:-1] = torch.tensor(tokens_list)
        if tokens.shape[0] < self.n_tokens * 2:
            pred_tokens = tokens[:tokens.shape[0] // 2]
            
            next_tokens = tokens[tokens.shape[0] // 2:]
            next_mask = torch.zeros(self.n_tokens, dtype=torch.int32)
            next_mask[:next_tokens.shape[0]] = 1
    
            n_pad = self.n_tokens - next_tokens.shape[0]
            next_tokens = torch.nn.functional.pad(next_tokens, (0, n_pad), 'constant', value=self.midi_encoder["PAD_None"])
        else:
            random_position = random.randint(0, tokens.shape[0] - self.n_tokens * 2)
            pred_tokens = tokens[random_position:random_position + self.n_tokens]
            next_tokens = tokens[random_position + self.n_tokens:random_position + self.n_tokens * 2]
            next_mask = torch.ones(self.n_tokens, dtype=torch.int32)
        return next_tokens, next_mask, pred_tokens
    
    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["duration"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)

        records_to_filter = exceeds_audio_length
        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "duration" in entry, (
                "Each dataset item should include field 'duration'"
                " - duration of audio (in seconds)."
            )
            assert "midi_path" in entry, (
                "Each dataset item should include field 'midi_path'" 
                " - path to midi file."
            )
