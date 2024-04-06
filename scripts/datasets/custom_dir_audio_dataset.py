import logging
from pathlib import Path

from scripts.datasets.custom_audio_dataset import CustomAudioDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, audio_dir, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix.lower() in [".mid", ".midi"]:
                entry["midi_path"] = str(path)
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
