import logging
from typing import List

from torch import int32, long, zeros

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate fields in dataset items
    """

    n_tokens = dataset_items[0]['input_ids'].shape[0]
    input_ids = zeros(
        len(dataset_items), 
        n_tokens,
        dtype=long
    )
    target_ids = zeros(
        len(dataset_items), 
        n_tokens,
        dtype=long
    )

    input_mask = zeros(
        len(dataset_items), 
        n_tokens,
        dtype=int32
    )

    target_mask = zeros(
        len(dataset_items), 
        n_tokens,
        dtype=int32
    )

    midi_path = []
    midi = []
    sequence_length = []

    for idx, item in enumerate(dataset_items):
        item_input_ids = item['input_ids']
        item_input_mask = item['input_mask']
        item_target_ids = item['target_ids']
        item_target_mask = item['target_mask']

        input_ids[idx, :] = item_input_ids
        input_mask[idx, :] = item_input_mask
        target_ids[idx, :] = item_target_ids
        target_mask[idx, :] = item_target_mask

        midi_path.append(item['midi_path'])
        midi.append(item['midi'])
        sequence_length.append(item['sequence_length'])

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "input_mask": input_mask,
        "target_mask": target_mask,
        "midi_path": midi_path,
        "midi": midi,
        "sequence_length": sequence_length,
    }
