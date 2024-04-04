import json
import logging
import shutil

import pandas as pd

from scripts.base.base_dataset import BaseDataset
from scripts.utils import ROOT_PATH, download

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip", 
}

PARTS = ['train', 'validation', 'test']

class MaestroDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if part not in PARTS:
            raise ValueError(f'Part {part} not in {PARTS}')
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "maestro-v2.0.0"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / "maestro-v2.0.0-midi.zip"
        print(f"Loading Maestro dataset")
        download(URL_LINKS["dataset"], str(arch_path))
        print('\nUnpacking files')
        shutil.unpack_archive(arch_path, self._data_dir)
        print('\n')
        for fpath in (self._data_dir / "maestro-v2.0.0").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        shutil.rmtree(str(self._data_dir / "maestro-v2.0.0"))


    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        labeling_path = self._data_dir / 'maestro-v2.0.0.csv'
        if not labeling_path.exists():
            self._load_dataset()

        print('Making index')
        data = pd.read_csv(labeling_path)
        data = data[data['split'] == part].copy()
        data = data.drop(columns=['split', 'audio_filename'])
        data = data.rename(columns={
            'canonical_composer': 'composer', 
            'canonical_title': 'title', 
            'midi_filename': 'midi_path'
        }) 
        data['midi_path'] = data['midi_path'].apply(lambda x: str(self._data_dir / x))
        index = data.to_dict(orient='records')

        return index
