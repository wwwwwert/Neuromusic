import logging
from typing import List

from torch import int32, zeros, long

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate fields in dataset items
    """

    n_tokens = dataset_items[0]['tokens'].shape[0]
    tokens = zeros(
        len(dataset_items), 
        n_tokens,
        dtype=long
    )

    tokens_mask = zeros(
        len(dataset_items), 
        n_tokens,
        dtype=int32
    )

    midi_path = []
    midi = []
    sequence_length = []
    target_sequences = []

    for idx, item in enumerate(dataset_items):
        item_tokens = item['tokens']
        item_mask = item['tokens_mask']

        tokens[idx, :] = item_tokens
        tokens_mask[idx, :] = item_mask
        midi_path.append(item['midi_path'])
        midi.append(item['midi'])
        sequence_length.append(item['sequence_length'])
        target_sequences.append(item_tokens[:item['sequence_length']].cpu().clone().detach())

    return {
        "input_ids": tokens,
        "midi_path": midi_path,
        "midi": midi,
        "padding_mask": tokens_mask,
        "sequence_length": sequence_length,
        "target_sequences": target_sequences
    }
