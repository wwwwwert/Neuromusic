from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import scripts.augmentations
import scripts.datasets
from scripts import batch_sampler as batch_sampler_module
from scripts.base.base_midi_encoder import BaseMiDiEncoder
from scripts.collate_fn.collate import collate_fn
from scripts.utils.parse_config import ConfigParser
import logging

logger = logging.getLogger(__name__)


def get_dataloaders(configs: ConfigParser, midi_encoder: BaseMiDiEncoder):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            midi_augs, tokens_augs = scripts.augmentations.from_configs(configs)
            drop_last = True
        else:
            midi_augs, tokens_augs = None, None
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(
                ds, scripts.datasets, midi_encoder=midi_encoder, config_parser=configs,
                midi_augs=midi_augs, tokens_augs=tokens_augs))
            ds_name = ds['type']
            ds_part = ds['args']['part']
            logger.info(f'{ds_name} {ds_part} contains {len(datasets[-1])} items')

        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        elif "batch_sampler" in params:
            batch_sampler = configs.init_obj(params["batch_sampler"], batch_sampler_module,
                                             data_source=dataset)
            bs, shuffle = 1, False
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=drop_last
        )
        dataloaders[split] = dataloader
    return dataloaders
