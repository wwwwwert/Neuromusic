import logging
from pathlib import Path

from scripts.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in data:
            assert len(data) > 0
            assert "midi_path" in entry
            assert Path(entry["midi_path"]).exists(), f"Path {entry['path']} doesn't exist"

        super().__init__(index, *args, **kwargs)
