import json
import logging
import shutil

import pandas as pd
from scripts.base.base_dataset import BaseDataset
from scripts.utils import ROOT_PATH, download
import pickle
from tqdm import tqdm
from scripts.utils import train_val_test_split
from symusic import Score


logger = logging.getLogger(__name__)

URL_LINKS = {
    "3-1": "https://huggingface.co/datasets/projectlosangeles/Los-Angeles-MIDI-Dataset/resolve/main/Los-Angeles-MIDI-Dataset-Ver-3-1-CC-BY-NC-SA.zip?download=true",
    "4-0": "https://huggingface.co/datasets/projectlosangeles/Los-Angeles-MIDI-Dataset/resolve/main/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA.zip?download=true"
}

PARTS = ['train', 'validation', 'test']

class LAMDataset(BaseDataset):
    def __init__(self, part, data_dir=None, version='3-1', *args, **kwargs):
        if part not in PARTS:
            raise ValueError(f'Part {part} not in {PARTS}')
        if version not in list(URL_LINKS.keys()):
            raise ValueError(f'Version {version} not in {list(URL_LINKS.keys())}')
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / f"Los-Angeles-MIDI-Dataset_{version}"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self.version = version
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / f"Los-Angeles-MIDI-Dataset_{self.version}.zip"
        print(f"Loading Los-Angeles-MIDI-Dataset")
        download(URL_LINKS[self.version], str(arch_path))
        print('\nUnpacking files')
        shutil.unpack_archive(arch_path, self._data_dir)
        print('\n')

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            splits_index = self._create_index()
            for index, split in zip(splits_index, PARTS):
                split_index_path = self._data_dir / f"{split}_index.json"
                with split_index_path.open("w") as f:
                    json.dump(index, f, indent=2)
            with index_path.open() as f:
                index = json.load(f)
        return index

    def _create_index(self):
        midis_path = self._data_dir / 'MIDIs'
        if not midis_path.exists():
            self._load_dataset()

        print('Making index')
        name_to_path = {}
        for p in midis_path.glob('**/*.mid'):
            name_to_path[p.stem] = str(p)

        pickle_path = self._data_dir / 'META_DATA' / 'LAMDa_META_DATA.pickle'
        print('Reading meta')
        with open(pickle_path, 'rb') as fp:
            data = pickle.load(fp, fix_imports=False)
        data = self.filter_readable_midis(data, name_to_path)
        splits = train_val_test_split(data, 0.6, 0.2, 0.2, random_state=0)
        splits_index = []
        for split, split_name in zip(splits, PARTS):
            split_idx = []
            for meta in tqdm(split, desc=f'indexing {split_name}'):
                features = {}
                name = meta[0]
                og_features = {k: v for k, v in meta[1][:17]}
                # after first 16 elements there are events of first 30 seconds
                features['midi_path'] = name_to_path[name]
                features['midi_ticks'] = og_features['midi_ticks']
                features['duration'] = og_features['pitches_times_sum_ms'] / 1000
                features['chords_density'] = og_features['total_number_of_chords_ms'] / og_features['total_number_of_chords']
                split_idx.append(features)
            splits_index.append(split_idx)

        return splits_index
    
    def filter_readable_midis(self, data, name_to_path):
        print('Filtering readable midis')
        n_damaged = 0
        new_data = []
        for meta in tqdm(data):
            name = meta[0]
            path = name_to_path[name]
            try:
                if not Score(path).empty():
                    new_data.append(meta)
            except:
                n_damaged += 1
        print('Damaged midis: ', n_damaged, '/', len(data))
        return new_data
