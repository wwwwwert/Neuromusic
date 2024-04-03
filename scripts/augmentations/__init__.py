from collections import Callable
from typing import List

import scripts.augmentations.midi_augmentations
import scripts.augmentations.tokens_augmentations
from scripts.augmentations.sequential import SequentialAugmentation
from scripts.utils.parse_config import ConfigParser


def from_configs(configs: ConfigParser):
    midi_augs = []
    if "augmentations" in configs.config and "midi" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["midi"]:
            midi_augs.append(
                configs.init_obj(aug_dict, scripts.augmentations.midi_augmentations)
            )

    tokens_augs = []
    if "augmentations" in configs.config and "tokens" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["tokens"]:
            tokens_augs.append(
                configs.init_obj(aug_dict, scripts.augmentations.tokens_augmentations)
            )
    return _to_function(midi_augs), _to_function(tokens_augs)


def _to_function(augs_list: List[Callable]):
    if len(augs_list) == 0:
        return None
    elif len(augs_list) == 1:
        return augs_list[0]
    else:
        return SequentialAugmentation(augs_list)
